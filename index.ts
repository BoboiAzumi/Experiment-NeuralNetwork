import { ReLu } from "./NeuralNetwork/Activation/ReLu";
import { Sigmoid } from "./NeuralNetwork/Activation/Sigmoid";
import { Layer } from "./NeuralNetwork/Layer";
import { BinaryCrossEntropy } from "./NeuralNetwork/LossFunction/BinaryCrossEntropy";


let data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
let output = [
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
]

let LossFunc = new BinaryCrossEntropy()
let layer = new Layer(2, 3, new ReLu(), LossFunc)
let layer2 = new Layer(3, 2, new Sigmoid(), LossFunc)

for(let epoch = 0; epoch < 10; epoch++){
    for(let i = 0; i < data.length; i++){
        let learning_rate = 0.01
        console.log(layer2.forward(layer.forward(data[i])))
        layer.backward(layer2.backward(output[i], learning_rate), learning_rate, true)
    }
}

console.log(layer2.forward(layer.forward(data[0])))