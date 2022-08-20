const brain = require("brain.js");
const data = require("./data.json");

const network = new brain.NeuralNetwork({
  binaryThresh: 0.5,
  hiddenLayers: [3],
  activation: "sigmoid",
  leakyReluAlpha: 0.01,
});

network.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
]);

const result = network.run([1, 1]);

console.log(
  `The result of the simple numerical dataset is ${(result * 1000).toFixed(0)}%`
);

// Text based dataset example

const lstmNetwork = new brain.recurrent.LSTM();

const trainingData = data.map(({ input, output }) => ({
  input,
  output,
}));

lstmNetwork.train(trainingData, {
  iterations: 2000, // This defaults to 20,000 but reducing it to boost performane for this demo
});

const testData = [
  "drive",
  "program had bugs",
  "code was tricky to follow",
  "repaired the power supply",
  "adjusted the monitor",
  "wrote unit tests",
];

const lstmResult = testData.map((testDataInput) =>
  lstmNetwork.run(testDataInput)
);

lstmResult.forEach((result, idx) =>
  console.log(
    `The LSTMNetwork thinks the input "${testData[idx]}" is catagorized as "${result}"`
  )
);
