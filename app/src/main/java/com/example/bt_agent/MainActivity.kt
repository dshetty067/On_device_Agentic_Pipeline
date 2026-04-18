package com.example.bt_agent

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
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
import okhttp3.OkHttpClient
import okhttp3.Request
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

data class ChatMessage(
    val id: Long,
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

    private var messageIdCounter = 0L
    private fun nextId() = ++messageIdCounter

    // ── JNI declarations ───────────────────────────────────────────────────
    external fun loadModel(path: String)
    external fun loadEmbeddingModel(path: String)   // ADD THIS LINE
    external fun generateText(query: String): String
    external fun setPdfPath(pdfPath: String, indexDir: String)

    // ── UI state ───────────────────────────────────────────────────────────
    private val isModelReady  = mutableStateOf(false)
    private val loadingStatus = mutableStateOf("Starting up...")
    private val isGenerating  = mutableStateOf(false)
    private val messages      = mutableStateListOf<ChatMessage>()
    private val streamingText = mutableStateOf("")
    private val currentStatus = mutableStateOf("")

    // ── PDF state ──────────────────────────────────────────────────────────
    private val pdfName       = mutableStateOf<String?>(null)   // null = nothing loaded yet

    // ── Callbacks called from native (JNI) ────────────────────────────────
    fun streamToken(piece: String) {
        runOnUiThread { streamingText.value += piece }
    }

    fun streamStatus(status: String) {
        runOnUiThread { currentStatus.value = status }
    }

    // FETCH WEATHER TOOL
    fun fetchWeather(query: String): String {
        android.util.Log.e("WEATHER", "fetchWeather CALLED query=$query thread=${Thread.currentThread().name}")

        var result = "No result found"
        val latch = java.util.concurrent.CountDownLatch(1)

        kotlinx.coroutines.GlobalScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            result = try {
                doFetchWeather(query)
            } catch (e: Exception) {
                android.util.Log.e("WEATHER", "Coroutine exception: ${e.javaClass.name}: ${e.message}", e)
                "Error: ${e.message}"
            } finally {
                latch.countDown()
            }
        }

        val completed = latch.await(15, java.util.concurrent.TimeUnit.SECONDS)
        if (!completed) android.util.Log.e("WEATHER", "TIMED OUT after 15s")
        android.util.Log.e("WEATHER", "fetchWeather RETURNING: $result")
        return result
    }

    private fun doFetchWeather(query: String): String {
        val client = okhttp3.OkHttpClient.Builder()
            .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
            .build()

        val q = query.lowercase()
        val cityGuess = extractCity(q) ?: return "Could not extract city from query."
        android.util.Log.d("WEATHER", "Extracted city: $cityGuess")

        val geoUrl = "https://geocoding-api.open-meteo.com/v1/search" +
                "?name=${java.net.URLEncoder.encode(cityGuess, "UTF-8")}&count=1&format=json"
        android.util.Log.d("WEATHER", "Geo URL: $geoUrl")

        val geoBody = client.newCall(
            okhttp3.Request.Builder().url(geoUrl).build()
        ).execute().use { it.body?.string() ?: "" }
        android.util.Log.d("WEATHER", "Geo raw: $geoBody")

        if (!geoBody.trim().startsWith("{")) return "Geocoding failed"

        val geoJson  = org.json.JSONObject(geoBody)
        val results  = geoJson.optJSONArray("results") ?: return "City not found: $cityGuess"
        val first    = results.getJSONObject(0)
        val lat      = first.getDouble("latitude")
        val lon      = first.getDouble("longitude")
        val cityName = first.optString("name")
        android.util.Log.d("WEATHER", "City: $cityName, lat: $lat, lon: $lon")

        val weatherUrl = "https://api.open-meteo.com/v1/forecast" +
                "?latitude=$lat&longitude=$lon" +
                "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weathercode,apparent_temperature" +
                "&timezone=auto"

        val wBody = client.newCall(
            okhttp3.Request.Builder().url(weatherUrl).build()
        ).execute().use { it.body?.string() ?: "" }

        if (!wBody.trim().startsWith("{")) return "Weather fetch failed"

        val wJson   = org.json.JSONObject(wBody)
        val current = wJson.optJSONObject("current") ?: return "No current weather data"

        val temp      = current.optDouble("temperature_2m")
        val feelsLike = current.optDouble("apparent_temperature")
        val humidity  = current.optDouble("relative_humidity_2m")
        val wind      = current.optDouble("wind_speed_10m")
        val code      = current.optInt("weathercode")

        val condition = when (code) {
            0          -> "Clear sky"
            1, 2, 3    -> "Partly cloudy"
            45, 48     -> "Foggy"
            51, 53, 55 -> "Drizzle"
            61, 63, 65 -> "Rain"
            71, 73, 75 -> "Snow"
            80, 81, 82 -> "Rain showers"
            95         -> "Thunderstorm"
            else       -> "Unknown"
        }

        return "Weather in $cityName: $condition, ${temp}°C (feels like ${feelsLike}°C), " +
                "Humidity: ${humidity.toInt()}%, Wind: ${wind} km/h"
    }

    private fun extractCity(query: String): String? {
        // grab word(s) after "in" or "at"
        val words = query.split(" ")
        val idx = words.indexOfFirst { it == "in" || it == "at" || it == "for" }
        if (idx != -1 && idx + 1 < words.size) {
            // take up to 2 words after "in/at/for" to handle "new york" etc
            return words.subList(idx + 1, minOf(idx + 3, words.size))
                .joinToString(" ")
                .replaceFirstChar { it.uppercase() }
        }
        return null
    }

    // WEB SEARCH TOOL - duckduckgo
    fun fetchWebSearch(query: String): String {
        android.util.Log.e("WEB_SEARCH", "fetchWebSearch CALLED query=$query")

        var result = "No result found"
        val latch = java.util.concurrent.CountDownLatch(1)

        kotlinx.coroutines.GlobalScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            result = try {
                doFetchWebSearch(query)
            } catch (e: Exception) {
                android.util.Log.e("WEB_SEARCH", "Exception: ${e.javaClass.name}: ${e.message}", e)
                "Web search error: ${e.message}"
            } finally {
                latch.countDown()
            }
        }

        val completed = latch.await(15, java.util.concurrent.TimeUnit.SECONDS)
        if (!completed) android.util.Log.e("WEB_SEARCH", "TIMED OUT after 15s")
        android.util.Log.e("WEB_SEARCH", "fetchWebSearch RETURNING: $result")
        return result
    }

    private fun doFetchWebSearch(query: String): String {
        val client = okhttp3.OkHttpClient.Builder()
            .connectTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(10, java.util.concurrent.TimeUnit.SECONDS)
            // DuckDuckGo requires a realistic User-Agent or it returns a CAPTCHA page
            .build()

        val encodedQuery = java.net.URLEncoder.encode(query, "UTF-8")
        // DuckDuckGo HTML endpoint — completely free, no API key
        val url = "https://html.duckduckgo.com/html/?q=$encodedQuery"
        android.util.Log.d("WEB_SEARCH", "DDG URL: $url")

        val request = okhttp3.Request.Builder()
            .url(url)
            .header("User-Agent",
                "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 " +
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36")
            .header("Accept-Language", "en-US,en;q=0.9")
            .build()

        val html = client.newCall(request).execute().use { it.body?.string() ?: "" }
        android.util.Log.d("WEB_SEARCH", "DDG response length: ${html.length}")

        if (html.isBlank()) return "No search results found"

        // ── Parse result snippets from DDG HTML ──────────────────────────────
        // DDG wraps each result in <div class="result__body"> with:
        //   <a class="result__a">TITLE</a>
        //   <a class="result__snippet">SNIPPET</a>
        val results = mutableListOf<String>()

        // Regex to extract title + snippet pairs
        val titleRegex  = Regex("""class="result__a"[^>]*>\s*(.*?)\s*</a>""", RegexOption.DOT_MATCHES_ALL)
        val snippetRegex = Regex("""class="result__snippet"[^>]*>\s*(.*?)\s*</a>""", RegexOption.DOT_MATCHES_ALL)

        val titles   = titleRegex.findAll(html).map  { it.groupValues[1].cleanHtml() }.toList()
        val snippets = snippetRegex.findAll(html).map { it.groupValues[1].cleanHtml() }.toList()

        val count = minOf(titles.size, snippets.size, 3) // top 3 results
        if (count == 0) return "No search results found"

        val sb = StringBuilder("Web search results for \"$query\":\n\n")
        for (i in 0 until count) {
            sb.append("${i + 1}. ${titles[i]}\n")
            sb.append("   ${snippets[i]}\n\n")
        }

        android.util.Log.d("WEB_SEARCH", "Parsed $count results")
        return sb.toString().trim()
    }

    // Strip HTML tags and decode common entities
    private fun String.cleanHtml(): String =
        this.replace(Regex("<[^>]+>"), "")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
            .replace("&nbsp;", " ")
            .replace(Regex("\\s+"), " ")
            .trim()


    // ── Lifecycle ──────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        thread {
            try {
                // ── Generation model ──────────────────────────────────────────
                runOnUiThread { loadingStatus.value = "Copying generation model..." }
                val genModel = File(filesDir, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
                if (!genModel.exists())
                    assets.open("models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf").use { i ->
                        FileOutputStream(genModel).use { o -> i.copyTo(o) }
                    }

                runOnUiThread { loadingStatus.value = "Loading generation model..." }
                loadModel(genModel.absolutePath)

                // ── Embedding model ───────────────────────────────────────────
                // CHANGE THIS FILENAME to match your actual .gguf in assets/models/
                runOnUiThread { loadingStatus.value = "Copying embedding model..." }
                val embModel = File(filesDir, "multilingual-e5-small-Q4_k_m.gguf")
                if (!embModel.exists())
                    assets.open("models/multilingual-e5-small-Q4_k_m.gguf").use { i ->
                        FileOutputStream(embModel).use { o -> i.copyTo(o) }
                    }

                runOnUiThread { loadingStatus.value = "Loading embedding model..." }
                loadEmbeddingModel(embModel.absolutePath)

                // ── Ready ─────────────────────────────────────────────────────
                runOnUiThread {
                    loadingStatus.value = "Ready"
                    isModelReady.value  = true
                    messages.add(ChatMessage(
                        id   = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "✅ Ready | Tools Available : Weather · Web Search· RAG · Book Flight"
                    ))
                }
            } catch (e: Exception) {
                runOnUiThread {
                    loadingStatus.value = "❌ ${e.message}"
                    messages.add(ChatMessage(
                        id   = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "Failed to load models: ${e.message}"
                    ))
                }
            }
        }

        setContent { BT_AgentTheme { ChatUI() } }
    }

    // ── Message send ───────────────────────────────────────────────────────
    private fun sendMessage(question: String) {
        messages.add(ChatMessage(id = nextId(), role = ChatMessage.Role.USER, text = question))
        isGenerating.value  = true
        streamingText.value = ""
        currentStatus.value = ""

        val assistantId = nextId()
        messages.add(ChatMessage(
            id          = assistantId,
            role        = ChatMessage.Role.ASSISTANT,
            text        = "",
            isStreaming = true
        ))

        thread {
            try {
                generateText(question)
                val finalText = streamingText.value
                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantId }
                    if (idx >= 0) messages[idx] = ChatMessage(
                        id   = assistantId,
                        role = ChatMessage.Role.ASSISTANT,
                        text = finalText.ifEmpty { "[No response]" }
                    )
                    isGenerating.value  = false
                    currentStatus.value = ""
                }
            } catch (e: Exception) {
                runOnUiThread {
                    val idx = messages.indexOfFirst { it.id == assistantId }
                    if (idx >= 0) messages[idx] = ChatMessage(
                        id   = assistantId,
                        role = ChatMessage.Role.ASSISTANT,
                        text = "❌ Error: ${e.message}"
                    )
                    isGenerating.value  = false
                    currentStatus.value = ""
                }
            }
        }
    }

    // ── PDF pick + copy ────────────────────────────────────────────────────
    // Copies the chosen PDF into internal storage so the native layer can
    // open it by a plain file-system path, then calls setPdfPath() via JNI.
    private fun onPdfPicked(uri: Uri?) {
        uri ?: return
        thread {
            try {
                val dest     = File(filesDir, "uploaded.pdf")
                val indexDir = File(filesDir, "rag_index").also { it.mkdirs() }

                // ── Wipe stale index so it rebuilds with correct embeddings ──
                listOf("hnsw_index.bin", "chunks.bin", "embeddings.bin").forEach {
                    File(indexDir, it).delete()
                }

                contentResolver.openInputStream(uri)!!.use { input ->
                    FileOutputStream(dest).use { out -> input.copyTo(out) }
                }

                val displayName = uri.lastPathSegment
                    ?.substringAfterLast('/')
                    ?.substringAfterLast(':')
                    ?: "document.pdf"

                setPdfPath(dest.absolutePath, indexDir.absolutePath)

                runOnUiThread {
                    pdfName.value = displayName
                    messages.add(ChatMessage(
                        id   = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "📄 Loaded: $displayName — ask anything about it"
                    ))
                }
            } catch (e: Exception) {
                runOnUiThread {
                    messages.add(ChatMessage(
                        id   = nextId(),
                        role = ChatMessage.Role.STATUS,
                        text = "❌ Failed to load PDF: ${e.message}"
                    ))
                }
            }
        }
    }

    // ── Colors ─────────────────────────────────────────────────────────────
    private val BgDark        = Color(0xFF0A0D14)
    private val SurfaceDark   = Color(0xFF131720)
    private val CardDark      = Color(0xFF1C2030)
    private val AccentBlue    = Color(0xFF4F8EF7)
    private val AccentPurple  = Color(0xFF9B6EF3)
    private val AccentGreen   = Color(0xFF43D9A0)
    private val AccentAmber   = Color(0xFFF7C34F)
    private val AccentOrange  = Color(0xFFFF8C42)
    private val UserBubble    = Color(0xFF1E3A6E)
    private val AiBubble      = Color(0xFF161B2E)
    private val TextPrimary   = Color(0xFFE4E8F4)
    private val TextSecondary = Color(0xFF6B7294)
    private val StatusColor   = Color(0xFF4ECDC4)
    private val BorderColor   = Color(0xFF252B3F)

    // ── Root composable ────────────────────────────────────────────────────
    @Composable
    fun ChatUI() {
        val modelReady by isModelReady
        val loading    by loadingStatus
        val generating by isGenerating
        val streaming  by streamingText
        val status     by currentStatus
        val currentPdf by pdfName

        var userInput by remember { mutableStateOf("") }
        val listState = rememberLazyListState()

        // PDF file-picker launcher
        val pdfLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.GetContent()
        ) { uri -> onPdfPicked(uri) }

        LaunchedEffect(messages.size, streaming) {
            if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
        }

        Box(modifier = Modifier.fillMaxSize().background(BgDark)) {
            Column(modifier = Modifier.fillMaxSize()) {

                TopBar(modelReady = modelReady, loading = loading, pdfName = currentPdf)

                LazyColumn(
                    state            = listState,
                    modifier         = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding   = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(messages, key = { it.id }) { msg ->
                        when (msg.role) {
                            ChatMessage.Role.USER      -> UserBubble(msg.text)
                            ChatMessage.Role.ASSISTANT -> AssistantBubble(
                                text        = if (msg.isStreaming) streaming else msg.text,
                                isStreaming = msg.isStreaming
                            )
                            ChatMessage.Role.STATUS    -> StatusBubble(msg.text)
                        }
                    }
                }

                // Inline status bar (tool progress)
                AnimatedVisibility(
                    visible = status.isNotEmpty(),
                    enter   = fadeIn() + expandVertically(),
                    exit    = fadeOut() + shrinkVertically()
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(Color(0xFF0A1020))
                            .padding(horizontal = 16.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text       = status,
                            fontSize   = 10.sp,
                            color      = StatusColor,
                            maxLines   = 1,
                            overflow   = TextOverflow.Ellipsis,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }

                InputBar(
                    value         = userInput,
                    onValueChange = { userInput = it },
                    onSend        = {
                        if (userInput.isNotBlank() && !generating) {
                            sendMessage(userInput.trim()); userInput = ""
                        }
                    },
                    onPickPdf     = { pdfLauncher.launch("application/pdf") },
                    enabled       = modelReady && !generating,
                    sendEnabled   = modelReady && userInput.isNotBlank() && !generating,
                    isGenerating  = generating,
                    hasPdf        = currentPdf != null
                )
            }
        }
    }

    // ── TopBar ─────────────────────────────────────────────────────────────
    @Composable
    fun TopBar(modelReady: Boolean, loading: String, pdfName: String?) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Brush.horizontalGradient(
                    colors = listOf(Color(0xFF0E1220), Color(0xFF151A2E))))
                .statusBarsPadding()
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            Row(
                verticalAlignment    = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier             = Modifier.fillMaxWidth()
            ) {
                Column {
                    Text(
                        "On-Device Agent",
                        fontSize     = 17.sp,
                        fontWeight   = FontWeight.Bold,
                        color        = TextPrimary,
                        letterSpacing = 0.5.sp
                    )
                    // Show loaded PDF name if present, else model status
                    Text(
                        text     = when {
                            pdfName != null -> "📄 $pdfName"
                            modelReady      -> "4 Tools available"
                            else            -> loading
                        },
                        fontSize = 11.sp,
                        color    = when {
                            pdfName != null -> AccentOrange
                            modelReady      -> AccentGreen
                            else            -> TextSecondary
                        },
                        maxLines  = 1,
                        overflow  = TextOverflow.Ellipsis,
                        modifier  = Modifier.widthIn(max = 220.dp)
                    )
                }
                Row(
                    verticalAlignment     = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (modelReady) {
                        Text(
                            "Weather · Web Search · Rag · Book Flight",
                            fontSize = 9.sp,
                            color    = AccentGreen,
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

    // ── Bubbles ────────────────────────────────────────────────────────────
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
            modifier          = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Start,
            verticalAlignment = Alignment.Top
        ) {
            Box(
                modifier = Modifier
                    .padding(top = 4.dp, end = 8.dp)
                    .size(26.dp)
                    .clip(CircleShape)
                    .background(Brush.linearGradient(listOf(AccentPurple, AccentBlue))),
                contentAlignment = Alignment.Center
            ) {
                Text("AI", fontSize = 8.sp, color = Color.White, fontWeight = FontWeight.Bold)
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
                    // Typing dots
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        repeat(3) { i ->
                            val alpha by rememberInfiniteTransition(label = "d$i")
                                .animateFloat(
                                    initialValue = 0.2f,
                                    targetValue  = 1f,
                                    animationSpec = infiniteRepeatable(
                                        animation  = tween(600, delayMillis = i * 180),
                                        repeatMode = RepeatMode.Reverse
                                    ),
                                    label = "a$i"
                                )
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(AccentBlue.copy(alpha = alpha))
                            )
                        }
                    }
                } else {
                    Text(
                        text       = if (isStreaming) "$text▌" else text,
                        fontSize   = 14.sp,
                        color      = TextPrimary,
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
                Text(text = text, fontSize = 12.sp, color = TextSecondary, lineHeight = 18.sp)
            }
        }
    }

    // ── InputBar ───────────────────────────────────────────────────────────
    @Composable
    fun InputBar(
        value: String,
        onValueChange: (String) -> Unit,
        onSend: () -> Unit,
        onPickPdf: () -> Unit,
        enabled: Boolean,
        sendEnabled: Boolean,
        isGenerating: Boolean,
        hasPdf: Boolean
    ) {
        Surface(color = SurfaceDark, shadowElevation = 8.dp) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .navigationBarsPadding()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    verticalAlignment     = Alignment.Bottom,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    // ── PDF attach button ──────────────────────────────────
                    IconButton(
                        onClick  = onPickPdf,
                        enabled  = enabled,
                        modifier = Modifier
                            .size(44.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (hasPdf)
                                    AccentOrange.copy(alpha = 0.18f)
                                else
                                    CardDark
                            )
                    ) {
                        Text(
                            text     = "📄",
                            fontSize = 18.sp
                        )
                    }

                    // ── Text field ─────────────────────────────────────────
                    TextField(
                        value         = value,
                        onValueChange = onValueChange,
                        modifier      = Modifier.weight(1f),
                        placeholder   = {
                            Text(
                                text     = if (hasPdf) "Ask about the document..." else "Ask me anything...",
                                fontSize = 13.sp,
                                color    = TextSecondary
                            )
                        },
                        enabled   = enabled,
                        maxLines  = 4,
                        colors    = TextFieldDefaults.colors(
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

                    // ── Send button ────────────────────────────────────────
                    IconButton(
                        onClick  = onSend,
                        enabled  = sendEnabled,
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
                                modifier    = Modifier.size(18.dp),
                                color       = Color.White,
                                strokeWidth = 2.dp
                            )
                        } else {
                            Icon(
                                imageVector     = Icons.Default.Send,
                                contentDescription = "Send",
                                tint            = if (sendEnabled) Color.White else TextSecondary,
                                modifier        = Modifier.size(18.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}