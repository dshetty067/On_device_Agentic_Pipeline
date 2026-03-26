package com.example.bt_agent

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.bt_agent.ui.theme.BT_AgentTheme
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

// ── Data model ────────────────────────────────────────────────────────────────
data class ChatMessage(
    val id: Long,
    val role: Role,
    val text: String,
    val isStreaming: Boolean = false,
    // metadata shown in expanded status card
    val plannerAction: String = "",
    val relevanceScore: Float = 0f
) {
    enum class Role { USER, ASSISTANT, STATUS }
}

// ── Routing label (extracted from status messages for the UI) ─────────────────
data class PipelineStep(val label: String, val done: Boolean, val active: Boolean)

class MainActivity : ComponentActivity() {

    companion object {
        init { System.loadLibrary("bt_agent_native") }
    }

    private var messageIdCounter = 0L
    private fun nextId() = ++messageIdCounter

    // ── JNI declarations ───────────────────────────────────────────────────
    external fun loadModel(path: String)
    external fun loadEmbeddingModel(path: String)
    external fun initToolRegistry()
    external fun generateText(pdfPath: String, storeDir: String, prompt: String): String

    // ── State ──────────────────────────────────────────────────────────────
    private val isModelReady    = mutableStateOf(false)
    private val loadingStatus   = mutableStateOf("Starting up...")
    private val isGenerating    = mutableStateOf(false)
    private val messages        = mutableStateListOf<ChatMessage>()
    private val streamingText   = mutableStateOf("")
    private val selectedPdfPath = mutableStateOf<String?>(null)
    private val selectedPdfName = mutableStateOf<String?>(null)
    private val currentStatus   = mutableStateOf("")
    private val pipelineSteps   = mutableStateListOf<PipelineStep>()

    // Pipeline step labels (in order) — kept in sync with agent_bt status msgs
    private val STEP_LABELS = listOf(
        "PDF Load", "Embed", "Retrieve", "Plan",
        "Tool Call", "Execute", "Answer", "Validate"
    )

    fun streamToken(piece: String) {
        runOnUiThread { streamingText.value += piece }
    }

    fun streamStatus(status: String) {
        runOnUiThread {
            currentStatus.value = status
            // Update pipeline steps by detecting node brackets like [3/8]
            val stepMatch = Regex("\\[(\\d+)/8]").find(status)
            if (stepMatch != null) {
                val stepNum = stepMatch.groupValues[1].toIntOrNull() ?: 0
                pipelineSteps.clear()
                for (i in STEP_LABELS.indices) {
                    val stepIdx = i + 1
                    pipelineSteps.add(PipelineStep(
                        label  = STEP_LABELS[i],
                        done   = stepIdx < stepNum,
                        active = stepIdx == stepNum
                    ))
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        thread {
            try {
                runOnUiThread { loadingStatus.value = "Copying models..." }

                val genModel = File(filesDir, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
                if (!genModel.exists())
                    copyAsset("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", genModel)

                val embModel = File(filesDir, "multilingual-e5-small-Q4_k_m.gguf")
                if (!embModel.exists())
                    copyAsset("models/multilingual-e5-small-Q4_k_m.gguf", embModel)

                runOnUiThread { loadingStatus.value = "Loading generation model..." }
                loadModel(genModel.absolutePath)

                runOnUiThread { loadingStatus.value = "Loading embedding model..." }
                loadEmbeddingModel(embModel.absolutePath)

                runOnUiThread { loadingStatus.value = "Initialising tools..." }
                initToolRegistry()

                runOnUiThread {
                    loadingStatus.value = "Ready"
                    isModelReady.value = true
                    messages.add(ChatMessage(
                        id = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "✅ Agent ready — 3 tools loaded\nAttach a PDF to begin."
                    ))
                }
            } catch (e: Exception) {
                runOnUiThread {
                    loadingStatus.value = "❌ ${e.message}"
                    messages.add(ChatMessage(
                        id = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "Failed to load models: ${e.message}"
                    ))
                }
            }
        }

        setContent { BT_AgentTheme { ChatUI() } }
    }

    private fun copyAsset(assetPath: String, dest: File) {
        assets.open(assetPath).use { i ->
            FileOutputStream(dest).use { o -> i.copyTo(o) }
        }
    }

    private fun copyPdfToInternal(uri: Uri): String {
        val name = getFileNameFromUri(uri) ?: "document.pdf"
        val dest = File(filesDir, "pdfs/$name")
        dest.parentFile?.mkdirs()
        contentResolver.openInputStream(uri)?.use { i ->
            FileOutputStream(dest).use { o -> i.copyTo(o) }
        }
        return dest.absolutePath
    }

    private fun getFileNameFromUri(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0) name = cursor.getString(idx)
            }
        }
        return name
    }

    private fun storeDir(pdfPath: String): String {
        val pdfName = File(pdfPath).nameWithoutExtension
        val dir = File(filesDir, "vector_stores/$pdfName")
        dir.mkdirs()
        return dir.absolutePath
    }

    private fun sendMessage(question: String) {
        val pdf = selectedPdfPath.value ?: return
        val store = storeDir(pdf)

        messages.add(ChatMessage(id = nextId(), role = ChatMessage.Role.USER, text = question))

        isGenerating.value = true
        streamingText.value = ""
        currentStatus.value = ""
        pipelineSteps.clear()

        val assistantId = nextId()
        messages.add(ChatMessage(
            id = assistantId,
            role = ChatMessage.Role.ASSISTANT,
            text = "",
            isStreaming = true
        ))

        thread {
            try {
                generateText(pdf, store, question)
                val finalText = streamingText.value

                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantId }
                    if (idx >= 0) {
                        messages[idx] = ChatMessage(
                            id = assistantId,
                            role = ChatMessage.Role.ASSISTANT,
                            text = finalText.ifEmpty { "[No response]" },
                            isStreaming = false
                        )
                    }
                    isGenerating.value = false
                    currentStatus.value = ""
                    pipelineSteps.clear()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantId }
                    if (idx >= 0) {
                        messages[idx] = ChatMessage(
                            id = assistantId,
                            role = ChatMessage.Role.ASSISTANT,
                            text = "❌ Error: ${e.message}",
                            isStreaming = false
                        )
                    }
                    isGenerating.value = false
                    currentStatus.value = ""
                    pipelineSteps.clear()
                }
            }
        }
    }

    // ── Color palette ──────────────────────────────────────────────────────
    private val BgDark        = Color(0xFF0A0D14)
    private val SurfaceDark   = Color(0xFF131720)
    private val CardDark      = Color(0xFF1C2030)
    private val AccentBlue    = Color(0xFF4F8EF7)
    private val AccentPurple  = Color(0xFF9B6EF3)
    private val AccentGreen   = Color(0xFF43D9A0)
    private val AccentAmber   = Color(0xFFF7C34F)
    private val UserBubble    = Color(0xFF1E3A6E)
    private val AiBubble      = Color(0xFF161B2E)
    private val TextPrimary   = Color(0xFFE4E8F4)
    private val TextSecondary = Color(0xFF6B7294)
    private val StatusColor   = Color(0xFF4ECDC4)
    private val BorderColor   = Color(0xFF252B3F)

    // ── Main UI ────────────────────────────────────────────────────────────
    @Composable
    fun ChatUI() {
        val modelReady  by isModelReady
        val loading     by loadingStatus
        val generating  by isGenerating
        val streaming   by streamingText
        val pdfPath     by selectedPdfPath
        val pdfName     by selectedPdfName
        val status      by currentStatus

        var userInput by remember { mutableStateOf("") }
        val listState = rememberLazyListState()

        LaunchedEffect(messages.size, streaming) {
            if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
        }

        val pdfLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    thread {
                        val name = getFileNameFromUri(uri) ?: "document.pdf"
                        val path = copyPdfToInternal(uri)
                        runOnUiThread {
                            selectedPdfPath.value = path
                            selectedPdfName.value = name
                            messages.clear()
                            messages.add(ChatMessage(
                                id = nextId(),
                                role = ChatMessage.Role.STATUS,
                                text = "📄 PDF loaded: $name\n" +
                                        "Vector store: ${
                                            if (File(storeDir(path), "vector_store.bin").exists())
                                                "cached ✓ (instant)" else "will build on first query"
                                        }\n\nTools available: search_pdf · calculate · summarise_document"
                            ))
                        }
                    }
                }
            }
        }

        Box(modifier = Modifier.fillMaxSize().background(BgDark)) {
            Column(modifier = Modifier.fillMaxSize()) {

                TopBar(modelReady = modelReady, loading = loading, pdfName = pdfName)

                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(messages, key = { it.id }) { msg ->
                        when (msg.role) {
                            ChatMessage.Role.USER      -> UserBubble(msg.text)
                            ChatMessage.Role.ASSISTANT -> AssistantBubble(
                                text = if (msg.isStreaming) streaming else msg.text,
                                isStreaming = msg.isStreaming
                            )
                            ChatMessage.Role.STATUS    -> StatusBubble(msg.text)
                        }
                    }
                }

                // ── Pipeline progress strip ────────────────────────────────
                AnimatedVisibility(
                    visible = generating && pipelineSteps.isNotEmpty(),
                    enter = fadeIn() + expandVertically(),
                    exit  = fadeOut() + shrinkVertically()
                ) {
                    PipelineStrip()
                }

                // ── Status ticker ──────────────────────────────────────────
                AnimatedVisibility(
                    visible = status.isNotEmpty(),
                    enter = fadeIn() + expandVertically(),
                    exit  = fadeOut() + shrinkVertically()
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0A1020))
                            .padding(horizontal = 16.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = status,
                            fontSize = 10.sp,
                            color = StatusColor,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }

                InputBar(
                    value = userInput,
                    onValueChange = { userInput = it },
                    onSend = {
                        if (userInput.isNotBlank() && pdfPath != null && !generating) {
                            sendMessage(userInput.trim()); userInput = ""
                        }
                    },
                    onAttach = {
                        pdfLauncher.launch(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                            addCategory(Intent.CATEGORY_OPENABLE); type = "application/pdf"
                        })
                    },
                    enabled = modelReady && !generating,
                    sendEnabled = modelReady && pdfPath != null && userInput.isNotBlank() && !generating,
                    pdfAttached = pdfPath != null,
                    isGenerating = generating
                )
            }
        }
    }

    // ── Pipeline Strip ─────────────────────────────────────────────────────
    @Composable
    fun PipelineStrip() {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF0D1022))
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                pipelineSteps.forEachIndexed { idx, step ->
                    val bgColor = when {
                        step.done   -> AccentGreen.copy(alpha = 0.2f)
                        step.active -> AccentBlue.copy(alpha = 0.3f)
                        else        -> Color(0xFF1A1F30)
                    }
                    val textColor = when {
                        step.done   -> AccentGreen
                        step.active -> AccentBlue
                        else        -> TextSecondary
                    }

                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(bgColor)
                            .padding(horizontal = 6.dp, vertical = 3.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = if (step.done) "✓ ${step.label}" else step.label,
                            fontSize = 9.sp,
                            color = textColor,
                            fontWeight = if (step.active) FontWeight.Bold else FontWeight.Normal
                        )
                    }

                    // Connector line
                    if (idx < pipelineSteps.size - 1) {
                        Text("›", fontSize = 9.sp, color = TextSecondary)
                    }
                }
            }
        }
    }

    @Composable
    fun TopBar(modelReady: Boolean, loading: String, pdfName: String?) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(Color(0xFF0E1220), Color(0xFF151A2E))
                    )
                )
                .statusBarsPadding()
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column {
                    Text(
                        text = "BT Agent",
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary,
                        letterSpacing = 0.5.sp
                    )
                    Text(
                        text = if (modelReady) {
                            pdfName?.let { "📄 $it" } ?: "No PDF — attach one to start"
                        } else loading,
                        fontSize = 11.sp,
                        color = if (modelReady && pdfName != null) AccentBlue else TextSecondary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                // Agent status indicator
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (modelReady) {
                        Text(
                            text = "3 tools",
                            fontSize = 9.sp,
                            color = AccentGreen,
                            modifier = Modifier
                                .clip(RoundedCornerShape(4.dp))
                                .background(AccentGreen.copy(alpha = 0.12f))
                                .padding(horizontal = 5.dp, vertical = 2.dp)
                        )
                    }
                    Box(
                        modifier = Modifier
                            .size(9.dp)
                            .clip(CircleShape)
                            .background(if (modelReady) AccentGreen else AccentAmber)
                    )
                }
            }
        }
    }

    @Composable
    fun UserBubble(text: String) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Box(
                modifier = Modifier
                    .widthIn(max = 290.dp)
                    .clip(RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp))
                    .background(Brush.linearGradient(listOf(AccentBlue, Color(0xFF3060C8))))
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                Text(text = text, fontSize = 14.sp, color = Color.White, lineHeight = 20.sp)
            }
        }
    }

    @Composable
    fun AssistantBubble(text: String, isStreaming: Boolean) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.Top
        ) {
            // AI avatar
            Box(
                modifier = Modifier
                    .padding(top = 4.dp, end = 8.dp)
                    .size(26.dp)
                    .clip(CircleShape)
                    .background(Brush.linearGradient(listOf(AccentPurple, AccentBlue))),
                contentAlignment = Alignment.Center
            ) {
                Text("BT", fontSize = 8.sp, color = Color.White, fontWeight = FontWeight.Bold)
            }

            Box(
                modifier = Modifier
                    .widthIn(max = 290.dp)
                    .clip(RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp))
                    .background(AiBubble)
                    .border(1.dp, BorderColor, RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                if (isStreaming && text.isEmpty()) {
                    // Loading dots
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        repeat(3) { i ->
                            val dotAlpha by rememberInfiniteTransition(label = "dot$i")
                                .animateFloat(
                                    initialValue = 0.2f, targetValue = 1f,
                                    animationSpec = infiniteRepeatable(
                                        animation = tween(600, delayMillis = i * 180),
                                        repeatMode = RepeatMode.Reverse
                                    ), label = "da$i"
                                )
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(AccentBlue.copy(alpha = dotAlpha))
                            )
                        }
                    }
                } else {
                    Text(
                        text = if (isStreaming) "$text▌" else text,
                        fontSize = 14.sp,
                        color = TextPrimary,
                        lineHeight = 21.sp
                    )
                }
            }
        }
    }

    @Composable
    fun StatusBubble(text: String) {
        Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xFF141929))
                    .border(1.dp, BorderColor, RoundedCornerShape(12.dp))
                    .padding(horizontal = 14.dp, vertical = 8.dp)
            ) {
                Text(
                    text = text,
                    fontSize = 12.sp,
                    color = TextSecondary,
                    lineHeight = 18.sp
                )
            }
        }
    }

    @Composable
    fun InputBar(
        value: String,
        onValueChange: (String) -> Unit,
        onSend: () -> Unit,
        onAttach: () -> Unit,
        enabled: Boolean,
        sendEnabled: Boolean,
        pdfAttached: Boolean,
        isGenerating: Boolean
    ) {
        Surface(color = SurfaceDark, shadowElevation = 8.dp) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                IconButton(
                    onClick = onAttach,
                    enabled = enabled,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(
                            if (pdfAttached) AccentBlue.copy(alpha = 0.2f)
                            else Color(0xFF1E2235)
                        )
                ) {
                    Icon(
                        imageVector = Icons.Default.AttachFile,
                        contentDescription = "Attach PDF",
                        tint = if (pdfAttached) AccentBlue else TextSecondary,
                        modifier = Modifier.size(20.dp)
                    )
                }

                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier.weight(1f),
                    placeholder = {
                        Text(
                            text = if (!pdfAttached) "Attach a PDF first..."
                            else "Ask anything about your document...",
                            fontSize = 13.sp,
                            color = TextSecondary
                        )
                    },
                    enabled = enabled && pdfAttached,
                    maxLines = 4,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor   = CardDark,
                        unfocusedContainerColor = CardDark,
                        disabledContainerColor  = Color(0xFF111520),
                        focusedTextColor        = TextPrimary,
                        unfocusedTextColor      = TextPrimary,
                        disabledTextColor       = TextSecondary,
                        focusedIndicatorColor   = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent,
                        disabledIndicatorColor  = Color.Transparent,
                        cursorColor             = AccentBlue
                    ),
                    shape = RoundedCornerShape(16.dp)
                )

                IconButton(
                    onClick = onSend,
                    enabled = sendEnabled,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(
                            if (sendEnabled)
                                Brush.linearGradient(listOf(AccentBlue, AccentPurple))
                            else
                                Brush.linearGradient(listOf(Color(0xFF1E2235), Color(0xFF1E2235)))
                        )
                ) {
                    if (isGenerating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = Color.White,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Send,
                            contentDescription = "Send",
                            tint = if (sendEnabled) Color.White else TextSecondary,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
            }
        }
    }
}