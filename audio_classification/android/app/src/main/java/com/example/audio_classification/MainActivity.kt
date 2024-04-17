package com.example.audio_classification

import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.BottomSheetScaffold
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.integerArrayResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AudioClassificationScreen()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AudioClassificationScreen(
    viewModel: MainViewModel = viewModel(
        factory = MainViewModel.getFactory(LocalContext.current.applicationContext)
    )
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            viewModel.startAudioClassification()
        } else {
            // Permission Denied
        }
    }

    DisposableEffect(Unit) {
        if (ContextCompat.checkSelfPermission(
                context, android.Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            viewModel.startAudioClassification()
        } else {
            launcher.launch(android.Manifest.permission.RECORD_AUDIO)
        }
        onDispose {
            viewModel.stopClassifier()
        }
    }
    BottomSheetScaffold(sheetContent = {
        BottomSheet(
            uiState = uiState,
            onModelSelected = {
                viewModel.setTFLiteModel(it)
            },
            onDelegateSelected = {
                viewModel.setDelegate(it)
            },
            onOverlapSelected = {
                viewModel.setOverlap(it.removeSuffix("%").toFloat() / 100)
            },
            onMaxResultSet = {
                viewModel.setMaxResults(it)
            },
            onThresholdSet = {
                viewModel.setThreshold(it)
            },
            onThreadsCountSet = {
                viewModel.setThreadCount(it)
            },
        )
    }) {
        ClassificationBody(uiState = uiState)
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun Header(modifier: Modifier = Modifier) {
    TopAppBar(
        modifier = modifier,
        title = {
            Image(
                modifier = Modifier.fillMaxSize(),
                painter = painterResource(id = R.drawable.tfl_logo),
                contentDescription = null,
            )

        },
    )
}

@Composable
fun BottomSheet(
    uiState: UiState,
    modifier: Modifier = Modifier,
    onModelSelected: (AudioClassificationHelper.TFLiteModel) -> Unit,
    onDelegateSelected: (option: AudioClassificationHelper.Delegate) -> Unit,
    onOverlapSelected: (option: String) -> Unit,
    onMaxResultSet: (value: Int) -> Unit,
    onThresholdSet: (value: Float) -> Unit,
    onThreadsCountSet: (value: Int) -> Unit,
) {
    val maxResults = uiState.setting.maxResults
    val threshold = uiState.setting.threshold
    val threadCount = uiState.setting.threadCount
    Column(modifier = modifier.padding(horizontal = 20.dp, vertical = 10.dp)) {
        Row {
            Text(modifier = Modifier.weight(0.5f), text = "Inference Time")
            Text(text = uiState.setting.inferenceTime.toString())
        }
        Spacer(modifier = Modifier.height(20.dp))
        ModelSelection(
            onModelSelected = onModelSelected,
        )
        MenuSample(label = "Delegate",
            options = AudioClassificationHelper.Delegate.entries.map { it.name }.toList(),
            onOptionSelected = {
                onDelegateSelected(AudioClassificationHelper.Delegate.valueOf(it))
            })
        MenuSample(
            label = "Overlap",
            options = listOf("25%", "50%", "75%", "100%"),
            onOptionSelected = onOverlapSelected
        )

        AdjustItem(
            name = "Max result", value = maxResults,
            onMinusClicked = {
                if (maxResults > 0) {
                    val max = maxResults - 1
                    onMaxResultSet(max)
                }
            },
            onPlusClicked = {
                if (maxResults < 5) {
                    val max = maxResults + 1
                    onMaxResultSet(max)
                }
            },
        )
        AdjustItem(
            name = "Threshold", value = threshold,
            onMinusClicked = {
                if (threshold > 0.1f) {
                    val newThreshold = (threshold - 0.1f).coerceAtLeast(0.1f)
                    onThresholdSet(newThreshold)
                }
            },
            onPlusClicked = {
                if (threshold < 1f) {
                    val newThreshold = threshold + 0.1f.coerceAtMost(1f)
                    onThresholdSet(newThreshold)
                }
            },
        )
        AdjustItem(
            name = "Threads", value = threadCount,
            onMinusClicked = {
                if (threadCount >= 2) {
                    val count = threadCount - 1
                    onThreadsCountSet(count)
                }
            },
            onPlusClicked = {
                if (threadCount < 5) {
                    val count = threadCount + 1
                    onThreadsCountSet(count)
                }
            },
        )
    }
}

@Composable
fun ClassificationBody(uiState: UiState, modifier: Modifier = Modifier) {
    val primaryProgressColorList = integerArrayResource(id = R.array.colors_progress_primary)
    val backgroundProgressColorList = integerArrayResource(id = R.array.colors_progress_background)
    Column(modifier = modifier) {
        Header()
        LazyColumn(
            contentPadding = PaddingValues(10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(uiState.classifications.size) { index ->
                val category = uiState.classifications[index]
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        modifier = Modifier.weight(0.4f),
                        text = category.label,
                        fontSize = 15.sp,
                    )
                    LinearProgressIndicator(
                        modifier = Modifier
                            .weight(0.6f)
                            .height(20.dp),
                        progress = category.score,
                        trackColor = Color(
                            backgroundProgressColorList[index % backgroundProgressColorList.size]
                        ),
                        color = Color(
                            primaryProgressColorList[index % backgroundProgressColorList.size]
                        ),
                    )
                }
            }
        }

    }
}

@Composable
fun MenuSample(
    label: String,
    modifier: Modifier = Modifier,
    options: List<String>,
    onOptionSelected: (option: String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    var option by remember { mutableStateOf(options.first()) }
    Row(
        modifier = modifier, verticalAlignment = Alignment.CenterVertically
    ) {
        Text(modifier = Modifier.weight(0.5f), text = label, fontSize = 15.sp)
        Text(text = option, fontSize = 15.sp)
        Box(
            modifier = Modifier.wrapContentSize(Alignment.TopStart)
        ) {
            IconButton(onClick = { expanded = true }) {
                Icon(Icons.Default.ArrowDropDown, contentDescription = "Localized description")
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                options.forEach {
                    DropdownMenuItem(text = {
                        Text(it, fontSize = 15.sp)
                    }, onClick = {
                        option = it
                        onOptionSelected(option)
                        expanded = false
                    })
                }
            }
        }
    }

}

@Composable
fun ModelSelection(
    modifier: Modifier = Modifier,
    onModelSelected: (AudioClassificationHelper.TFLiteModel) -> Unit,
) {
    val radioOptions = AudioClassificationHelper.TFLiteModel.entries.map { it.name }.toList()
    var selectedOption by remember { mutableStateOf(radioOptions.first()) }

    Column(modifier = modifier) {
        radioOptions.forEach { option ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                RadioButton(selected = (option == selectedOption), onClick = {
                    onModelSelected(AudioClassificationHelper.TFLiteModel.valueOf(option))
                    selectedOption = option
                } // Recommended for accessibility with screen readers
                )
                Text(modifier = Modifier.padding(start = 16.dp), text = option, fontSize = 15.sp)
            }
        }
    }
}

@Composable
fun AdjustItem(
    name: String,
    value: Number,
    modifier: Modifier = Modifier,
    onMinusClicked: () -> Unit,
    onPlusClicked: () -> Unit,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            modifier = Modifier.weight(0.5f),
            text = name,
            fontSize = 15.sp,
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = {
                onMinusClicked()
            }) {
                Text(text = "-", fontSize = 15.sp)
            }
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                modifier = Modifier.width(30.dp),
                textAlign = TextAlign.Center,
                text = if (value is Float) String.format(
                    Locale.US, "%.1f", value
                ) else value.toString(),
                fontSize = 15.sp,
            )
            Spacer(modifier = Modifier.width(10.dp))
            Button(onClick = {
                onPlusClicked()
            }) {
                Text(text = "+", fontSize = 15.sp)
            }
        }
    }
}
