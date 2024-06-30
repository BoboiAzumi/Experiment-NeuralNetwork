import { ReLu } from "./NeuralNetwork/Activation/ReLu";
import { Sigmoid } from "./NeuralNetwork/Activation/Sigmoid";
import { DenseLayer } from "./NeuralNetwork/DenseLayer";
import type { Layer } from "./NeuralNetwork/Layer";
import { BinaryCrossEntropy } from "./NeuralNetwork/LossFunction/BinaryCrossEntropy";
import { Dataset } from "./dataset";

const relu = new ReLu()
const sigmoid = new Sigmoid()
const binarycrossentropy = new BinaryCrossEntropy()

const dataset = new Dataset()
dataset.load()
dataset.parse()

const data = dataset.get()
const output = dataset.getBinaryLabel(0)

let layer: Layer[] = []
let outputlayer: number[][] = []
let outputbackward: number[][] = []
let learning_rate = 0.01

layer[0] = new DenseLayer(4, 8, relu, binarycrossentropy)
layer[1] = new DenseLayer(8, 1, sigmoid, binarycrossentropy)

// Train
for(let epoch = 0; epoch < 40; epoch++){
    for(let i = 0; i < data.y.length; i++){
        outputlayer[0] = layer[0].forward(data.x[i])
        outputlayer[1] = layer[1].forward(outputlayer[0])

        let loss = binarycrossentropy.loss(outputlayer[1], output[i])

        outputbackward[0] = layer[1].backward(output[i], learning_rate)
        outputbackward[2] = layer[0].backward(outputbackward[0], learning_rate, true)
    }
}

let accuracy = 0
let sample = data.x.length

for(let i = 0; i < sample; i++){

    outputlayer[0] = layer[0].forward(data.x[i])
    outputlayer[1] = layer[1].forward(outputlayer[0])

    console.log("Csv Index : "+i)
    console.log("Predict : "+ (outputlayer[1][0] < 0.00001 ? 0 : outputlayer[1]) )
    console.log("Actual : "+output[i])

    let eq = (outputlayer[1][0] >= 0.5 ? 1 : 0)
    accuracy += eq == output[i][0] ? 1 : 0
}

accuracy = accuracy / sample

console.log ("\nAccuracy : "+accuracy*100)