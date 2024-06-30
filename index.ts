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
    [1, 0]
]

let LossFunc = new BinaryCrossEntropy()
let layer = new Layer(2, 6, new ReLu(), LossFunc)
let layer2 = new Layer(6, 2, new Sigmoid(), LossFunc)
let learning_rate = 0.125

for(let epoch = 0; epoch < 100; epoch++){
    for(let i = 0; i < data.length; i++){
        let o = layer2.forward(layer.forward(data[i]))
        let loss = LossFunc.loss(o, output[i])
        console.log("Step : "+i)
        console.log("Epoch : "+epoch)
        console.log("Loss : "+loss)
        console.log("")
        layer.backward(layer2.backward(output[i], learning_rate),learning_rate, true)
        if(loss < 0.5){
            epoch = 250
            i = 4
        }
    }
}

let o = layer2.forward(layer.forward(data[1]))
console.log(o)