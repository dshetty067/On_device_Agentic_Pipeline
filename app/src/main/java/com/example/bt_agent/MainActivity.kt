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
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.bt_agent.ui.theme.BT_AgentTheme
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

// ── Data model ────────────────────────────────────────────────────────────────
// ── Data model ────────────────────────────────────────────────────────────────
data class ChatMessage(
    val id: Long,   // ✅ FIXED (removed default timestamp)
    val role: Role,
    val text: String,
    val isStreaming: Boolean = false
) {
    enum class Role { USER, ASSISTANT, STATUS }
}

class MainActivity : ComponentActivity() {

    companion object {
        init { System.loadLibrary("bt_agent_native") }
    }

    // ✅ UNIQUE ID GENERATOR (NEW)
    private var messageIdCounter = 0L
    private fun generateUniqueId(): Long {
        return ++messageIdCounter
    }

    external fun loadModel(path: String)
    external fun loadEmbeddingModel(path: String)
    external fun generateText(pdfPath: String, storeDir: String, prompt: String): String

    // ── State ──────────────────────────────────────────────────────────────
    private val isModelReady      = mutableStateOf(false)
    private val loadingStatus     = mutableStateOf("Starting up...")
    private val isGenerating      = mutableStateOf(false)
    private val messages          = mutableStateListOf<ChatMessage>()
    private val streamingText     = mutableStateOf("")
    private val selectedPdfPath   = mutableStateOf<String?>(null)
    private val selectedPdfName   = mutableStateOf<String?>(null)
    private val currentStatus     = mutableStateOf("")

    // JNI callbacks
    fun streamToken(piece: String) {
        runOnUiThread { streamingText.value += piece }
    }
    fun streamStatus(status: String) {
        runOnUiThread { currentStatus.value = status }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        thread {
            try {
                runOnUiThread { loadingStatus.value = "Copying models..." }

                val genModel = File(filesDir, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")
                if (!genModel.exists()) copyAsset("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", genModel)

                val embModel = File(filesDir, "multilingual-e5-small-Q4_k_m.gguf")
                if (!embModel.exists()) copyAsset("models/multilingual-e5-small-Q4_k_m.gguf", embModel)

                runOnUiThread { loadingStatus.value = "Loading generation model..." }
                loadModel(genModel.absolutePath)

                runOnUiThread { loadingStatus.value = "Loading embedding model..." }
                loadEmbeddingModel(embModel.absolutePath)

                runOnUiThread {
                    loadingStatus.value = "Ready"
                    isModelReady.value = true
                    messages.add(ChatMessage(
                        id = generateUniqueId(),   // ✅ FIX
                        role = ChatMessage.Role.STATUS,
                        text = "Models loaded. Attach a PDF and start asking questions."
                    ))
                }
            } catch (e: Exception) {
                runOnUiThread {
                    loadingStatus.value = "❌ ${e.message}"
                    messages.add(ChatMessage(
                        id = generateUniqueId(),   // ✅ FIX
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

        // ✅ USER MESSAGE FIX
        messages.add(ChatMessage(
            id = generateUniqueId(),
            role = ChatMessage.Role.USER,
            text = question
        ))

        isGenerating.value = true
        streamingText.value = ""
        currentStatus.value = ""

        // ✅ ASSISTANT ID FIX
        val assistantMsgId = generateUniqueId()

        messages.add(ChatMessage(
            id = assistantMsgId,
            role = ChatMessage.Role.ASSISTANT,
            text = "",
            isStreaming = true
        ))

        thread {
            try {
                generateText(pdf, store, question)
                val finalText = streamingText.value

                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantMsgId }
                    if (idx >= 0) {
                        messages[idx] = ChatMessage(
                            id = assistantMsgId,
                            role = ChatMessage.Role.ASSISTANT,
                            text = finalText.ifEmpty { "[No response]" },
                            isStreaming = false
                        )
                    }
                    isGenerating.value = false
                    currentStatus.value = ""
                }
            } catch (e: Exception) {
                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantMsgId }
                    if (idx >= 0) {
                        messages[idx] = ChatMessage(
                            id = assistantMsgId,
                            role = ChatMessage.Role.ASSISTANT,
                            text = "❌ Error: ${e.message}",
                            isStreaming = false
                        )
                    }
                    isGenerating.value = false
                    currentStatus.value = ""
                }
            }
        }
    }

    // ── Color palette ──────────────────────────────────────────────────────
    private val BgDark       = Color(0xFF0F1117)
    private val SurfaceDark  = Color(0xFF1A1D27)
    private val CardDark     = Color(0xFF21253A)
    private val AccentBlue   = Color(0xFF4F8EF7)
    private val AccentPurple = Color(0xFF9B6EF3)
    private val UserBubble   = Color(0xFF2A3A5C)
    private val AiBubble     = Color(0xFF1E2235)
    private val TextPrimary  = Color(0xFFE8ECF4)
    private val TextSecondary= Color(0xFF8B91A8)
    private val StatusColor  = Color(0xFF4ECDC4)

    // ── Main UI ────────────────────────────────────────────────────────────
    @Composable
    fun ChatUI() {
        val modelReady    by isModelReady
        val loading       by loadingStatus
        val generating    by isGenerating
        val streaming     by streamingText
        val pdfPath       by selectedPdfPath
        val pdfName       by selectedPdfName
        val status        by currentStatus

        var userInput by remember { mutableStateOf("") }
        val listState = rememberLazyListState()

        // Auto-scroll when messages change or streaming updates
        LaunchedEffect(messages.size, streaming) {
            if (messages.isNotEmpty()) {
                listState.animateScrollToItem(messages.size - 1)
            }
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
                            // Clear old chat when new PDF is loaded
                            messages.clear()
                            messages.add(ChatMessage(
                                id = generateUniqueId(),   // ✅ FIX
                                role = ChatMessage.Role.STATUS,
                                text = "📄 PDF loaded: $name\nVector store: ${
                                    if (File(storeDir(path), "hnsw_index.bin").exists())
                                        "cached ✓" else "will build on first query"
                                }"
                            ))
                        }
                    }
                }
            }
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(BgDark)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {

                // ── Top bar ────────────────────────────────────────────────
                TopBar(
                    modelReady = modelReady,
                    loading = loading,
                    pdfName = pdfName
                )

                // ── Chat messages ──────────────────────────────────────────
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
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

                // ── Status ticker ──────────────────────────────────────────
                AnimatedVisibility(
                    visible = status.isNotEmpty(),
                    enter = fadeIn() + expandVertically(),
                    exit  = fadeOut() + shrinkVertically()
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0D1520))
                            .padding(horizontal = 16.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = status,
                            fontSize = 11.sp,
                            color = StatusColor,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }

                // ── Input bar ──────────────────────────────────────────────
                InputBar(
                    value = userInput,
                    onValueChange = { userInput = it },
                    onSend = {
                        if (userInput.isNotBlank() && pdfPath != null && !generating) {
                            sendMessage(userInput.trim())
                            userInput = ""
                        }
                    },
                    onAttach = {
                        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                            addCategory(Intent.CATEGORY_OPENABLE)
                            type = "application/pdf"
                        }
                        pdfLauncher.launch(intent)
                    },
                    enabled = modelReady && !generating,
                    sendEnabled = modelReady && pdfPath != null && userInput.isNotBlank() && !generating,
                    pdfAttached = pdfPath != null,
                    isGenerating = generating
                )
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
                        colors = listOf(Color(0xFF141829), Color(0xFF1A1D2E))
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
                        text = "PDF Assistant",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color = TextPrimary,
                        letterSpacing = 0.3.sp
                    )
                    Text(
                        text = if (modelReady) {
                            pdfName?.let { "📄 $it" } ?: "No PDF selected"
                        } else loading,
                        fontSize = 11.sp,
                        color = if (modelReady && pdfName != null) AccentBlue else TextSecondary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                // Model status dot
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(
                            if (modelReady) Color(0xFF4CAF50) else Color(0xFFFF9800)
                        )
                )
            }
        }
    }

    @Composable
    fun UserBubble(text: String) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            Box(
                modifier = Modifier
                    .widthIn(max = 300.dp)
                    .clip(RoundedCornerShape(18.dp, 4.dp, 18.dp, 18.dp))
                    .background(
                        Brush.linearGradient(
                            colors = listOf(AccentBlue, Color(0xFF3A6FD8))
                        )
                    )
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                Text(
                    text = text,
                    fontSize = 14.sp,
                    color = Color.White,
                    lineHeight = 20.sp
                )
            }
        }
    }

    @Composable
    fun AssistantBubble(text: String, isStreaming: Boolean) {
        // Blinking cursor animation
        val cursorAlpha by rememberInfiniteTransition(label = "cursor").animateFloat(
            initialValue = 1f, targetValue = 0f,
            animationSpec = infiniteRepeatable(
                animation = tween(500, easing = LinearEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "blink"
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.Top
        ) {
            // AI avatar
            Box(
                modifier = Modifier
                    .padding(top = 4.dp, end = 8.dp)
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.linearGradient(
                            colors = listOf(AccentPurple, AccentBlue)
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Text("AI", fontSize = 9.sp, color = Color.White, fontWeight = FontWeight.Bold)
            }

            Box(
                modifier = Modifier
                    .widthIn(max = 300.dp)
                    .clip(RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp))
                    .background(AiBubble)
                    .border(
                        width = 1.dp,
                        color = Color(0xFF2A2F45),
                        shape = RoundedCornerShape(4.dp, 18.dp, 18.dp, 18.dp)
                    )
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                if (isStreaming && text.isEmpty()) {
                    // Loading dots
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        repeat(3) { i ->
                            val dotAlpha by rememberInfiniteTransition(label = "dot$i").animateFloat(
                                initialValue = 0.3f, targetValue = 1f,
                                animationSpec = infiniteRepeatable(
                                    animation = tween(600, delayMillis = i * 150),
                                    repeatMode = RepeatMode.Reverse
                                ),
                                label = "dotAnim$i"
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
        Box(
            modifier = Modifier.fillMaxWidth(),
            contentAlignment = Alignment.Center
        ) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(Color(0xFF1A2030))
                    .border(1.dp, Color(0xFF252A40), RoundedCornerShape(12.dp))
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
        Surface(
            color = SurfaceDark,
            shadowElevation = 8.dp
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Attach PDF button
                IconButton(
                    onClick = onAttach,
                    enabled = enabled,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(
                            if (pdfAttached) AccentBlue.copy(alpha = 0.2f)
                            else Color(0xFF252A3A)
                        )
                ) {
                    Icon(
                        imageVector = Icons.Default.AttachFile,
                        contentDescription = "Attach PDF",
                        tint = if (pdfAttached) AccentBlue else TextSecondary,
                        modifier = Modifier.size(20.dp)
                    )
                }

                // Text field
                TextField(
                    value = value,
                    onValueChange = onValueChange,
                    modifier = Modifier.weight(1f),
                    placeholder = {
                        Text(
                            text = if (!pdfAttached) "Attach a PDF first..." else "Ask about your PDF...",
                            fontSize = 14.sp,
                            color = TextSecondary
                        )
                    },
                    enabled = enabled && pdfAttached,
                    maxLines = 4,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor   = CardDark,
                        unfocusedContainerColor = CardDark,
                        disabledContainerColor  = Color(0xFF16192A),
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

                // Send button
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
                                Brush.linearGradient(listOf(Color(0xFF252A3A), Color(0xFF252A3A)))
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