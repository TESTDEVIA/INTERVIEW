function obtenerValorParametro(nombre) {
  var urlParams = new URLSearchParams(window.location.search);
  console.log(nombre,"=",urlParams.get(nombre));
  return urlParams.get(nombre);
}

function obtenerDatosBinance() {
  return new Promise(function(resolve, reject) {
    var timestampActual = Math.floor(Date.now() / 1000);
    var timestamp24HorasAtras = timestampActual - (48 * 60 * 60);

    var url = 'https://api.binance.com/api/v3/klines';
    var params = {
      symbol: globalSymbol || 'BTCBUSD',
      interval: '15m',
      startTime: timestamp24HorasAtras * 1000,
      endTime: timestampActual * 1000
    };

    var request = new XMLHttpRequest();
    request.open('GET', url + '?' + serializeParams(params), true);

    request.onload = function() {
      if (request.status >= 200 && request.status < 400) {
        var data = JSON.parse(request.responseText);
        var registros = [];

        for (var i = 0; i < data.length; i++) {
          var timestamp = data[i][0];
          var precio = data[i][4];

          registros.push({
            timestamp: timestamp,
            precio: precio
          });
        }

        var datosJSON = JSON.stringify(registros, null, 2);
        console.log('Obtener los datos de Binance: OK');
        localStorage.setItem('datosBinance', datosJSON);
        resolve(registros);
      } else {
        reject(new Error('Error al obtener los datos de Binance:', request.statusText));
      }
    };

    request.onerror = function() {
      reject(new Error('Error de conexión'));
    };

    request.send();
  });
}

function serializeParams(params) {
  var query = [];
  for (var key in params) {
    query.push(encodeURIComponent(key) + '=' + encodeURIComponent(params[key]));
  }
  return query.join('&');
}

async function getData() {
  try {
    const binanceData = await obtenerDatosBinance();

    // Obtener el mínimo timestamp
    // const minTimestamp = Math.min(...binanceData.map(crypto => crypto.timestamp));

    const cleaned = binanceData.map(crypto => ({
      timestamp: crypto.timestamp/1000,
      price: Number(crypto.precio)
    })).filter(crypto => (crypto.price != null && crypto.timestamp != null));

    return cleaned;
  } catch (error) {
    console.error(error);
    return null;
  }
}

function createModel() {
  const model = tf.sequential();

  // Capa LSTM (Long Short-Term Memory)
  model.add(tf.layers.lstm({ units: 32, returnSequences: true, inputShape: [1] }));

  // Capas ocultas adicionales (opcional)
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // Capa de salida
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}



/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.timestamp)
    const labels = data.map(d => d.price);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = globalEpochs || 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Rendimiento del Entrenamiento' },
      ['mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map((d) => ({
    x: d.timestamp,
    y: d.price,
  }));

  const minValue = Math.min(...values.map((v) => v.y));
  // values.forEach((v) => (v.y -= minValue));

  const symbol = globalSymbol || 'BTCBUSD';
  const chartTitle = `Model Predictions vs Original Data (${symbol})`;

  const subtitle = `Actual Trend: calculating...`;

  tfvis.render.linechart(
    { name: chartTitle, subtitle: subtitle },
    { values: values },
    {
      xLabel: 'timestamp',
      yLabel: 'price',
      height: 300,
      zoomToFit: true,
      yValueLabel: (index) => values[index].y.toFixed(2),
    }
  );

  // Create the model
  const model = createModel();
  // tfvis.show.modelSummary({ name: 'BTCBUSD Prediction' }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  console.log('Entrenamiento iniciado. Espere por favor');
  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Entrenamiento finalizado');

  // Make some predictions using the model and compare them to the
  // original data

  let trend = testModel(model, data, tensorData);

  // Generate new timestamps
  const newTimestamps = [];
  for (let i = 0; i < 5; i++) {
    const predictionTimestamp = Math.floor(Date.now()) + i * 15 * 60 * 1000;
    newTimestamps.push(Math.floor(predictionTimestamp / 1000));
  }

  // Make predictions for the new timestamps
  // testModelFuture(model, data, tensorData, newTimestamps, trend);  
  const { minPrice, maxPrice } = await testModelFuture(model, data, tensorData, newTimestamps, trend);

}

function calculateSlope(data) {
  const n = data.length;
  let sumXY = 0;
  let sumX = 0;
  let sumY = 0;
  let sumX2 = 0;

  for (let i = 0; i < n; i++) {
    const { x, y } = data[i];
    sumXY += x * y;
    sumX += x;
    sumY += y;
    sumX2 += x * x;
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  return slope;
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.timestamp, y: d.price,
  }));

  const last50PredictedPrices = predictedPoints.slice(-50);

  // Uso:
  const slope = calculateSlope(last50PredictedPrices);
  console.log('Slope:', slope);
  
  const actualTrend = slope > 0 ? 'UP' : 'DW';

  console.log('Slope:', slope);
  console.log('Actual trend:', actualTrend);

  const symbol = globalSymbol || 'BTCBUSD';
  // const chartTitle = `Predictions vs Original (${symbol}) Trend (${actualTrend})`;
  const chartTitle = `Model Predictions vs Original Data (${symbol})`;

  // Construye la clave utilizando el formato "localstorage://<symbol>"
  const key = `localstorage://${symbol}`;
  // Guarda actualTrend en el Local Storage
  localStorage.setItem(key, actualTrend);

  const subtitle = `Actual Trend: ${actualTrend}`;

  tfvis.render.linechart(
    { name: chartTitle, subtitle: subtitle },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'timestamp',
      yLabel: 'price',
      height: 300,
      zoomToFit: true,
    }
  );

  return (actualTrend);
}

function findLocalExtrema(data) {
  const extrema = [];

  for (let i = 1; i < data.length - 1; i++) {
    const prevPrice = data[i - 1].price;
    const currPrice = data[i].price;
    const nextPrice = data[i + 1].price;

    if (currPrice < prevPrice && currPrice < nextPrice) {
      // Mínimo local
      extrema.push({
        timestamp: data[i].timestamp,
        price: currPrice,
        type: 'minimum'
      });
    } else if (currPrice > prevPrice && currPrice > nextPrice) {
      // Máximo local
      extrema.push({
        timestamp: data[i].timestamp,
        price: currPrice,
        type: 'maximum'
      });
    }
  }

  return extrema;
}

async function testModelFuture(model, inputData, normalizationData, newTimestamps, actualTrend) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const tensorData = convertToTensor(inputData);
  const { inputs } = tensorData;

  const newInputs = tf.tidy(() => {
    const newInputTensor = tf.tensor2d(newTimestamps, [newTimestamps.length, 1]);
    const normalizedInputs = newInputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    return normalizedInputs;
  });

  const preds = model.predict(newInputs);
  const unNormPreds = tf.tidy(() => {
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
    return unNormPreds.dataSync();
  });

  for (let i = 0; i < newTimestamps.length; i++) {
    const predictionTimestamp = newTimestamps[i];
    const predictedPrice = unNormPreds[i];

    inputData.push({
      timestamp: predictionTimestamp,
      price: predictedPrice
    });
  }

  const predictedPoints = inputData.map(d => ({
    x: d.timestamp,
    y: d.price
  }));

  const originalPoints = inputData.filter(d => !newTimestamps.includes(d.timestamp)).map(d => ({
    x: d.timestamp,
    y: d.price
  }));

  const last50PredictedPrices = predictedPoints.slice(-50);

  const symbol = globalSymbol || 'BTCBUSD';

  const lastTimestamp = inputData[inputData.length - 1].timestamp;
  // const newTimestamps = Array.from({ length: 4 }, (_, i) => lastTimestamp + (i + 1) * 15 * 60 * 1000);

  console.log("Future projections")

  const predictions = predictedPoints.slice(-4);  

  let futurePoints = [];

  predictions.forEach((prediction) => {
    const timestamp = prediction.x;
    const price = prediction.y.toFixed(5);
    const date = new Date(timestamp*1000).toLocaleString("es-VE", { timeZone: "America/Caracas" });

    futurePoints.push({timestamp: timestamp, date: date, price: price});
    console.log(`Fecha/Hora: ${date}`);
    console.log(`Precio: ${price}`);
  });

  var minPrice = Math.min(...inputData.map((d) => d.price));
  var maxPrice = Math.max(...inputData.map((d) => d.price));

  const priceDiff = maxPrice - minPrice;
  var percentageDiff = (priceDiff / minPrice) * 100;

  const minPriceTimestamp = originalPoints.find(d => d.y === minPrice).x;
  const maxPriceTimestamp = originalPoints.find(d => d.y === maxPrice).x;

  var buydate = new Date(minPriceTimestamp*1000).toLocaleString("es-VE", { timeZone: "America/Caracas" });
  var selldate = new Date(maxPriceTimestamp*1000).toLocaleString("es-VE", { timeZone: "America/Caracas" });

  console.log(`Absolute Minimum Price:  ${minPrice.toFixed(4)} at ${buydate}`);
  console.log(`Absolute Maximum Price:  ${maxPrice.toFixed(4)} at ${selldate}`);

  if (minPriceTimestamp > maxPriceTimestamp) {
    percentageDiff = - percentageDiff;
  }
  console.log(`Absolute Maximum Difference: ${percentageDiff.toFixed(3)}%`);

  const extrema = findLocalExtrema(inputData);
  let lastMaxTimestamp = -Infinity;
  let lastMinTimestamp = -Infinity;
  minPrice = Infinity;
  maxPrice = -Infinity;
  let maxPercentageDiff = -Infinity;

  const trades = [];

  for (const point of extrema) {
    if (point.type === 'minimum') {
      lastMinTimestamp = point.timestamp;
      minPrice = point.price;
    } else if (point.type === 'maximum') {
      lastMaxTimestamp = point.timestamp;
      maxPrice = point.price;
    }
    if (lastMaxTimestamp > lastMinTimestamp) {
      const percentageDiff = ((maxPrice - minPrice) / minPrice) * 100;
      buydate = new Date(lastMinTimestamp*1000).toLocaleString("es-VE", { timeZone: "America/Caracas" });
      selldate = new Date(lastMaxTimestamp*1000).toLocaleString("es-VE", { timeZone: "America/Caracas" });
      console.log("------------ Trade ------------");
      console.log(`Buy:   ${minPrice.toFixed(4)} at ${buydate}`);
      console.log(`Sell:  ${maxPrice.toFixed(4)} at ${selldate}`);
      console.log(`Fee:   ${percentageDiff.toFixed(2)}%`);

      trades.push({
        label: "buy",
        timestamp: lastMinTimestamp,
        price: minPrice
      });      
      trades.push({
        label: "sell",
        timestamp: lastMaxTimestamp,
        price: maxPrice
      });      

      if (percentageDiff > maxPercentageDiff) {
        maxPercentageDiff = percentageDiff;
      }
    }
  }

console.log("-------------------------------");
console.log(`Fee:   ${maxPercentageDiff.toFixed(2)}% (Optimal)`);

const data = {
  symbol: symbol,
  actualTrend: actualTrend,
  predictions: futurePoints,
  minPrice: minPrice,
  maxPrice: maxPrice,
  percentageDiff: percentageDiff.toFixed(3),
  optimalFee: maxPercentageDiff.toFixed(3)
};



return {minPrice, minPrice};

}


document.addEventListener('DOMContentLoaded', run);