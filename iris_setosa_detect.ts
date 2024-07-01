import type { NeuralNetwork } from "./NeuralNetwork/NeuralNetwork";
import { ReLu } from "./NeuralNetwork/Activation/ReLu";
import { Sigmoid } from "./NeuralNetwork/Activation/Sigmoid";
import { DenseLayer } from "./NeuralNetwork/Layer/DenseLayer";
import type { Layer } from "./NeuralNetwork/Layer/Layer";
import { BinaryCrossEntropy } from "./NeuralNetwork/LossFunction/BinaryCrossEntropy";
import { Dataset } from "./dataset";
import type { LossFunction } from "./NeuralNetwork/LossFunction/LossFunction";


export class IrisSetosaDetect implements NeuralNetwork{
    learning_rate: number = 0.0175
    epoch: number = 75
    LossFunc: LossFunction = new BinaryCrossEntropy()
    dataset: Dataset = new Dataset()
    layer: Layer[] = []
    epoch_history: number[] = []
    loss_history: number[] = []
    accuracy_history: number[] = []

    constructor(){
        this.dataset.load()
        this.dataset.parse()
        this.setup()
        this.train()
    }

    setup(){
        const relu = new ReLu()
        const sigmoid = new Sigmoid()
        
        this.layer[0] = new DenseLayer(4, 8, relu, this.LossFunc)
        this.layer[1] = new DenseLayer(8, 1, sigmoid, this.LossFunc)
    }

    train(){
        let outputlayer: number[][] = []
        let outputbackward: number[][] = []
        const data = this.dataset.get()
        const output = this.dataset.getBinaryLabel(0)
        for(let epochs = 0; epochs < this.epoch; epochs++){
            let loss = 0
            let accuracy = 0
            /* Training Model */
            for(let i = 0; i < data.x.length; i+=2){
                outputlayer[0] = this.layer[0].forward(data.x[i])
                outputlayer[1] = this.layer[1].forward(outputlayer[0])
                outputbackward[0] = this.layer[1].backward(output[i], this.learning_rate)
                outputbackward[1] = this.layer[0].backward(outputbackward[0], this.learning_rate, true)
            }
            /* Testing Model */
            for(let i = 0; i < data.y.length; i++){
                outputlayer[0] = this.layer[0].forward(data.x[i])
                outputlayer[1] = this.layer[1].forward(outputlayer[0])
                loss += this.LossFunc.loss(outputlayer[1], output[i])
                let eq = (outputlayer[1][0] >= 0.8 ? 1 : 0)
                accuracy += eq == output[i][0] ? 1 : 0
            }

            accuracy = accuracy / data.y.length
            loss = loss / data.y.length

            this.epoch_history.push(epochs+1)
            this.loss_history.push(loss)
            this.accuracy_history.push(accuracy)
            
            console.log("Epoch    : "+(epochs+1))
            console.log("Avg Loss : "+loss)
            console.log("Accuracy : "+accuracy)
            console.log("")
        }

        /* Final Testing */
        for(let i = 0; i < data.y.length; i++){
            outputlayer[0] = this.layer[0].forward(data.x[i])
            outputlayer[1] = this.layer[1].forward(outputlayer[0])
            console.log("")
            console.log("Actual : "+output[i])
            console.log("Predict : "+outputlayer[1])
        }
    }

    get_history(){
        return {
            epoch: this.epoch_history,
            loss: this.loss_history,
            accuracy: this.accuracy_history
        }
    }

    predict(input: number[]){
        let outputlayer: number[][] = []
        let transform_input = this.dataset.transform(input)
        outputlayer[0] = this.layer[0].forward(transform_input)
        outputlayer[1] = this.layer[1].forward(outputlayer[0])

        return outputlayer[1]
    }
}
