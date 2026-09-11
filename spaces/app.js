const state = {
  files: [],
  labels: [],
  model: null,
  onnxSession: null,
  onnxLabels: [],
  stop: false,
  history: { loss: [], accuracy: [] }
};

const $ = (id) => document.getElementById(id);
const imageExtensions = /\.(png|jpe?g|gif|bmp|webp)$/i;

function log(message) {
  $('log').textContent = message;
}

function setStatus(message) {
  $('trainingStatus').textContent = message;
}

function drawChart() {
  const canvas = $('metricsChart');
  const context = canvas.getContext('2d');
  const width = canvas.clientWidth * window.devicePixelRatio;
  const height = 280 * window.devicePixelRatio;
  canvas.width = width;
  canvas.height = height;
  context.clearRect(0, 0, width, height);
  context.scale(window.devicePixelRatio, window.devicePixelRatio);
  const chartWidth = canvas.clientWidth;
  const chartHeight = 280;
  context.strokeStyle = '#d9ded5';
  context.lineWidth = 1;
  for (let row = 1; row < 5; row += 1) {
    const y = row * chartHeight / 5;
    context.beginPath(); context.moveTo(0, y); context.lineTo(chartWidth, y); context.stroke();
  }
  const drawLine = (values, color) => {
    if (!values.length) return;
    const max = Math.max(1, ...values);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    values.forEach((value, index) => {
      const x = values.length === 1 ? 0 : index * chartWidth / (values.length - 1);
      const y = chartHeight - (value / max) * (chartHeight - 18) - 8;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  };
  drawLine(state.history.loss, '#ef744d');
  drawLine(state.history.accuracy, '#0c7761');
}

function summarizeDataset() {
  const counts = {};
  state.files.forEach(({ label }) => { counts[label] = (counts[label] || 0) + 1; });
  $('datasetSummary').classList.remove('empty');
  $('datasetSummary').textContent = `${state.files.length} images / ${state.labels.length} classes\n${state.labels.map((label) => `${label}: ${counts[label]}`).join('  |  ')}`;
  $('trainButton').disabled = state.files.length < 2 || state.labels.length < 2;
  setStatus(state.labels.length >= 2 ? 'Ready to train' : 'Need two classes');
}

$('datasetInput').addEventListener('change', (event) => {
  state.files = [...event.target.files]
    .filter((file) => imageExtensions.test(file.name))
    .map((file) => {
      const parts = (file.webkitRelativePath || file.name).split('/');
      return { file, label: parts.length > 1 ? parts[parts.length - 2] : 'default' };
    });
  state.labels = [...new Set(state.files.map(({ label }) => label))].sort();
  summarizeDataset();
  log('Dataset loaded locally. Nothing was uploaded.');
});

async function imageTensor(file) {
  const bitmap = await createImageBitmap(file);
  const tensor = tf.tidy(() => tf.image.resizeBilinear(tf.browser.fromPixels(bitmap), [128, 128]).toFloat().div(127.5).sub(1));
  bitmap.close();
  return tensor;
}

async function makeTrainingTensors() {
  const images = [];
  const labels = [];
  for (const item of state.files) {
    images.push(await imageTensor(item.file));
    labels.push(state.labels.indexOf(item.label));
  }
  const xs = tf.stack(images);
  images.forEach((tensor) => tensor.dispose());
  return { xs, ys: tf.oneHot(tf.tensor1d(labels, 'int32'), state.labels.length) };
}

function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [128, 128, 3], filters: 16, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dropout({ rate: 0.35 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: state.labels.length, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(Number($('learningRate').value)), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

$('trainButton').addEventListener('click', async () => {
  $('trainButton').disabled = true;
  $('stopButton').disabled = false;
  state.stop = false;
  state.history = { loss: [], accuracy: [] };
  try {
    setStatus('Preparing tensors');
    log('Decoding images in this browser...');
    const tensors = await makeTrainingTensors();
    state.model?.dispose();
    state.model = buildModel();
    const epochs = Number($('epochs').value);
    setStatus('Training locally');
    await state.model.fit(tensors.xs, tensors.ys, {
      epochs,
      batchSize: Number($('batchSize').value),
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, metrics) => {
          const loss = Number(metrics.loss);
          const accuracy = Number(metrics.acc ?? metrics.accuracy ?? 0);
          state.history.loss.push(loss);
          state.history.accuracy.push(accuracy);
          $('epochMetric').textContent = `${epoch + 1} / ${epochs}`;
          $('lossMetric').textContent = loss.toFixed(4);
          $('accuracyMetric').textContent = `${(accuracy * 100).toFixed(1)}%`;
          drawChart();
          log(`Epoch ${epoch + 1} complete on ${tf.getBackend()}.`);
          await tf.nextFrame();
          if (state.stop) state.model.stopTraining = true;
        }
      }
    });
    tensors.xs.dispose(); tensors.ys.dispose();
    setStatus(state.stop ? 'Training stopped' : 'Training complete');
    log(state.stop ? 'Training stopped by the user.' : 'Model is ready for local inference.');
    $('downloadButton').disabled = false;
  } catch (error) {
    setStatus('Training failed');
    log(error.message);
  } finally {
    $('stopButton').disabled = true;
    $('trainButton').disabled = false;
  }
});

$('stopButton').addEventListener('click', () => { state.stop = true; });

async function predictWithBrowserModel(file) {
  const tensor = await imageTensor(file);
  const result = state.model.predict(tensor.expandDims(0));
  const probabilities = await result.data();
  tensor.dispose(); result.dispose();
  const index = probabilities.indexOf(Math.max(...probabilities));
  $('prediction').classList.remove('empty');
  $('prediction').textContent = `${state.labels[index]} — ${(probabilities[index] * 100).toFixed(1)}% confidence`;
}

$('imageInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    if (state.model) await predictWithBrowserModel(file);
    else if (state.onnxSession) await predictWithOnnx(file);
    else $('prediction').textContent = 'Train a browser model or load an ONNX model first.';
  } catch (error) { $('prediction').textContent = `Inference failed: ${error.message}`; }
});

$('downloadButton').addEventListener('click', () => state.model?.save('downloads://image-classifier-local'));

$('labelsInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) state.onnxLabels = JSON.parse(await file.text());
});

$('onnxInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    state.onnxSession = await ort.InferenceSession.create(await file.arrayBuffer());
    $('onnxStatus').classList.remove('empty');
    $('onnxStatus').textContent = `Loaded ${file.name}. Select an image to run inference.`;
  } catch (error) { $('onnxStatus').textContent = `ONNX load failed: ${error.message}`; }
});

async function predictWithOnnx(file) {
  const tensor = await imageTensor(file);
  const input = new ort.Tensor('float32', Float32Array.from(await tensor.transpose([2, 0, 1]).expandDims(0).data()), [1, 3, 128, 128]);
  const inputName = state.onnxSession.inputNames[0];
  const output = await state.onnxSession.run({ [inputName]: input });
  const scores = Object.values(output)[0].data;
  const index = scores.indexOf(Math.max(...scores));
  $('prediction').classList.remove('empty');
  $('prediction').textContent = `${state.onnxLabels[index] || `class ${index}`} — ONNX inference complete`;
  tensor.dispose();
}

(async () => {
  try {
    if (navigator.gpu) await tf.setBackend('webgpu');
    await tf.ready();
    $('runtimeStatus').textContent = `Browser runtime: ${tf.getBackend().toUpperCase()} / data stays local`;
  } catch (error) {
    await tf.setBackend('webgl'); await tf.ready();
    $('runtimeStatus').textContent = `Browser runtime: ${tf.getBackend().toUpperCase()} fallback / data stays local`;
  }
  drawChart();
})();

window.addEventListener('resize', drawChart);
