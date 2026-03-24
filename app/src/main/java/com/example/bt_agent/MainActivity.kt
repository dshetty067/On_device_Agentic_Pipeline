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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.bt_agent.ui.theme.BT_AgentTheme
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    companion object {
        init { System.loadLibrary("bt_agent_native") }
    }

    external fun loadModel(path: String)
    external fun loadEmbeddingModel(path: String)
    external fun generateText(pdfPath: String, storeDir: String, prompt: String): String


    // ── State ──────────────────────────────────────────────────────────────
    private val isModelReady       = mutableStateOf(false)
    private val loadingStatus      = mutableStateOf("Starting up...")
    private val logs               = mutableStateListOf<Pair<String, LogLevel>>()
    private val streamingResponse  = mutableStateOf("")
    private val nodeStatus         = mutableStateOf("")
    private val selectedPdfPath    = mutableStateOf<String?>(null)
    private val selectedPdfName    = mutableStateOf("No PDF selected")

    enum class LogLevel { INFO, SUCCESS, WARNING, ERROR }

    private fun addLog(message: String, level: LogLevel = LogLevel.INFO) {
        val ts = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
            .format(java.util.Date())
        runOnUiThread { logs.add("[$ts] $message" to level) }
    }

    // JNI callbacks
    fun streamToken(piece: String)  { runOnUiThread { streamingResponse.value += piece } }
    fun streamStatus(status: String){ runOnUiThread { nodeStatus.value = status } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        thread {
            try {
                addLog("App started")
                runOnUiThread { loadingStatus.value = "Copying models to storage..." }

                // ── Generation model ───────────────────────────────────────
                val genModel = File(filesDir, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

                if (genModel.exists()) {
                    addLog("Gen model cached (${genModel.length()/1024/1024} MB)", LogLevel.INFO)
                } else {
                    addLog("Copying generation model from assets...", LogLevel.WARNING)
                    copyAsset("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", genModel)
                }

                // ── Embedding model ────────────────────────────────────────
                val embModel = File(filesDir, "multilingual-e5-small-Q4_k_m.gguf")
                if (embModel.exists()) {
                    addLog("Embedding model cached (${embModel.length()/1024/1024} MB)", LogLevel.INFO)
                } else {
                    addLog("Copying E5 embedding model from assets...", LogLevel.WARNING)
                    copyAsset("models/multilingual-e5-small-Q4_k_m.gguf", embModel)
                }

                // ── Load both models ───────────────────────────────────────
                runOnUiThread { loadingStatus.value = "Loading generation model..." }
                addLog("Loading generation model into RAM...", LogLevel.INFO)
                val t1 = System.currentTimeMillis()
                loadModel(genModel.absolutePath)
                addLog("Generation model loaded in ${(System.currentTimeMillis()-t1)/1000}s ✓", LogLevel.SUCCESS)

                runOnUiThread { loadingStatus.value = "Loading embedding model..." }
                addLog("Loading embedding model into RAM...", LogLevel.INFO)
                val t2 = System.currentTimeMillis()
                loadEmbeddingModel(embModel.absolutePath)
                addLog("Embedding model loaded in ${(System.currentTimeMillis()-t2)/1000}s ✓", LogLevel.SUCCESS)

                runOnUiThread {
                    loadingStatus.value = "✓ Ready — select a PDF to begin"
                    isModelReady.value = true
                }

            } catch (e: Exception) {
                addLog("FATAL: ${e.message}", LogLevel.ERROR)
                runOnUiThread { loadingStatus.value = "❌ ${e.message}" }
            }
        }

        setContent { BT_AgentTheme { ChatScreen() } }
    }

    private fun copyAsset(assetPath: String, dest: File) {
        assets.open(assetPath).use { i ->
            FileOutputStream(dest).use { o -> i.copyTo(o) }
        }
        addLog("Copied $assetPath (${dest.length()/1024/1024} MB)", LogLevel.SUCCESS)
    }

    // Copy PDF from content URI to app's internal storage so C++ can read it
    private fun copyPdfToInternal(uri: Uri): String {
        val name = getFileNameFromUri(uri) ?: "document.pdf"
        val dest = File(filesDir, "pdfs/$name")
        dest.parentFile?.mkdirs()
        contentResolver.openInputStream(uri)?.use { i ->
            FileOutputStream(dest).use { o -> i.copyTo(o) }
        }
        addLog("PDF copied to internal: ${dest.absolutePath}", LogLevel.SUCCESS)
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

    // ── Vector store directory for this PDF ────────────────────────────────
    private fun storeDir(pdfPath: String): String {
        val pdfName = File(pdfPath).nameWithoutExtension
        val dir = File(filesDir, "vector_stores/$pdfName")
        dir.mkdirs()
        return dir.absolutePath
    }

    // ── UI ─────────────────────────────────────────────────────────────────
    @Composable
    fun ChatScreen() {
        var userInput          by remember { mutableStateOf("") }
        val modelReady         by isModelReady
        val status             by loadingStatus
        val streaming          by streamingResponse
        val currentNodeStatus  by nodeStatus
        val pdfName            by selectedPdfName
        val pdfPath            by selectedPdfPath

        val logScrollState      = rememberScrollState()
        val responseScrollState = rememberScrollState()

        LaunchedEffect(logs.size)  { logScrollState.animateScrollTo(logScrollState.maxValue) }
        LaunchedEffect(streaming)  { responseScrollState.animateScrollTo(responseScrollState.maxValue) }

        // PDF picker launcher
        val pdfLauncher = rememberLauncherForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    thread {
                        val name = getFileNameFromUri(uri) ?: "document.pdf"
                        addLog("PDF selected: $name", LogLevel.INFO)
                        runOnUiThread { selectedPdfName.value = name }
                        val path = copyPdfToInternal(uri)
                        runOnUiThread { selectedPdfPath.value = path }
                        addLog("PDF ready at: $path", LogLevel.SUCCESS)
                    }
                }
            }
        }

        Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
            Column(
                modifier = Modifier
                    .padding(padding)
                    .padding(16.dp)
                    .fillMaxSize()
            ) {

                // ── Status Banner ──────────────────────────────────────────
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (modelReady) Color(0xFF1B5E20)
                        else MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        if (!modelReady) {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Spacer(Modifier.height(6.dp))
                        }
                        Text(
                            text = status, fontSize = 13.sp,
                            color = if (modelReady) Color.White
                            else MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                Spacer(Modifier.height(10.dp))

                // ── PDF Picker ─────────────────────────────────────────────
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (pdfPath != null) Color(0xFF1A237E)
                        else MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Row(
                        modifier = Modifier
                            .padding(10.dp)
                            .fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "📄 $pdfName",
                                fontSize = 12.sp,
                                color = if (pdfPath != null) Color.White
                                else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            if (pdfPath != null) {
                                Text(
                                    text = "Vector store: ${File(storeDir(pdfPath!!), "hnsw_index.bin").exists().let { if (it) "cached ✓" else "will build" }}",
                                    fontSize = 10.sp,
                                    color = Color(0xFF90CAF9)
                                )
                            }
                        }
                        Button(
                            onClick = {
                                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                                    addCategory(Intent.CATEGORY_OPENABLE)
                                    type = "application/pdf"
                                }
                                pdfLauncher.launch(intent)
                            },
                            enabled = modelReady
                        ) {
                            Text("Pick PDF", fontSize = 12.sp)
                        }
                    }
                }

                Spacer(Modifier.height(10.dp))

                // ── Log Console ────────────────────────────────────────────
                Text("Logs:", style = MaterialTheme.typography.labelMedium)
                Spacer(Modifier.height(4.dp))
                Card(
                    modifier = Modifier.fillMaxWidth().height(130.dp),
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A1A))
                ) {
                    Column(
                        modifier = Modifier.padding(8.dp).verticalScroll(logScrollState)
                    ) {
                        if (logs.isEmpty()) Text("Waiting...", fontSize = 11.sp, color = Color.Gray)
                        logs.forEach { (msg, level) ->
                            val color = when (level) {
                                LogLevel.SUCCESS -> Color(0xFF66BB6A)
                                LogLevel.ERROR   -> Color(0xFFEF5350)
                                LogLevel.WARNING -> Color(0xFFFFCA28)
                                LogLevel.INFO    -> Color(0xFFB0BEC5)
                            }
                            Text(text = msg, fontSize = 11.sp, color = color)
                        }
                    }
                }

                Spacer(Modifier.height(10.dp))

                // ── Node Status ────────────────────────────────────────────
                if (currentNodeStatus.isNotEmpty()) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(containerColor = Color(0xFF263238))
                    ) {
                        Text(
                            text = currentNodeStatus,
                            fontSize = 12.sp,
                            color = Color(0xFF80CBC4),
                            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                        )
                    }
                    Spacer(Modifier.height(6.dp))
                }

                // ── Response Area ──────────────────────────────────────────
                Text("Response:", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Card(
                    modifier = Modifier.fillMaxWidth().weight(1f)
                ) {
                    Box(
                        modifier = Modifier
                            .padding(10.dp)
                            .verticalScroll(responseScrollState)
                    ) {
                        Text(
                            text = streaming.ifEmpty { "Response will appear here..." },
                            fontSize = 14.sp,
                            color = if (streaming.isEmpty())
                                MaterialTheme.colorScheme.onSurfaceVariant
                            else MaterialTheme.colorScheme.onSurface
                        )
                    }
                }

                Spacer(Modifier.height(10.dp))

                // ── Input ──────────────────────────────────────────────────
                TextField(
                    value = userInput,
                    onValueChange = { userInput = it },
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = { Text("Ask something about your PDF...") },
                    enabled = modelReady && pdfPath != null
                )

                Spacer(Modifier.height(8.dp))

                Button(
                    onClick = {
                        val question = userInput
                        val pdf      = pdfPath ?: return@Button
                        val store    = storeDir(pdf)

                        streamingResponse.value = ""
                        nodeStatus.value        = ""

                        addLog("Query: \"${question.take(50)}\"", LogLevel.INFO)
                        addLog("PDF:   $pdf",                      LogLevel.INFO)
                        addLog("Store: $store",                    LogLevel.INFO)

                        thread {
                            try {
                                val start = System.currentTimeMillis()
                                val result = generateText(pdf, store, question)
                                val elapsed = (System.currentTimeMillis() - start) / 1000
                                addLog("Done in ${elapsed}s (${result.length} chars)", LogLevel.SUCCESS)
                            } catch (e: Exception) {
                                addLog("ERROR: ${e.message}", LogLevel.ERROR)
                                runOnUiThread {
                                    streamingResponse.value = "❌ ${e.message}"
                                }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled  = modelReady && pdfPath != null && userInput.isNotBlank()
                ) {
                    Text(when {
                        !modelReady       -> "Loading models..."
                        pdfPath == null   -> "Select a PDF first"
                        else              -> "Send"
                    })
                }
            }
        }
    }
}