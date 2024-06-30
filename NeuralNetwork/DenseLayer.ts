import type { Activation } from "./Activation/Activation"
import type { Layer } from "./Layer"
import type { LossFunction } from "./LossFunction/LossFunction"

export class DenseLayer implements Layer{
    bias: number[]
    weight: number[][]
    sum: number[] = []
    input: number[] = []
    output: number[] = []
    input_size: number
    neuron_size: number
    activation: Activation
    loss_function: LossFunction
    
    constructor(input_size: number, neuron_size: number, activation: Activation, loss_function: LossFunction){
        this.input_size = input_size
        this.neuron_size = neuron_size
        this.bias = []
        this.weight = []
        this.activation = activation
        this.loss_function = loss_function

        for(let i = 0; i < neuron_size; i++){
            this.bias.push(1)
        }

        for(let i = 0; i < neuron_size; i++){
            let neuron: number[] = []
            for(let j = 0; j < input_size; j++){
                neuron.push(1)
            }
            this.weight.push(neuron)
        }
    }

    printMatrix(){
        console.log("Weight : ")
        for(let i = 0; i < this.weight.length; i++){
            console.log(this.weight[i])
        }
        console.log("\nBias : ")
        console.log(this.bias)
    }

    printWeight(){
        for(let i = 0; i < this.weight.length; i++){
            console.log(this.weight[i])
        }
    }

    printIO(){
        console.log("=============================================================")
        console.log("Input : ")
        console.log(this.input)
        console.log("Weight : ")
        console.log(this.weight)
        console.log("Bias : ")
        console.log(this.bias)
        console.log("Sum : ")
        console.log(this.sum)
        console.log("Output : ")
        console.log(this.output)
        console.log("=============================================================")
    }

    forward(x: number[]){
        this.input = x
        this.sum = []
        this.output = []
        for(let i = 0; i < this.neuron_size; i++){
            let sum: number = 0
            for(let j = 0; j < this.input_size; j++){
                sum += x[j] * this.weight[i][j]
            }
            sum += this.bias[i]
            this.sum.push(sum)
            this.output.push(this.activation.func(sum))
        }

        return this.output
    }

    backward(y: number[], learning_rate: number, is_hidden: boolean = false) {
        let loss_out_gradient: number[] = []
        let derivative_activation = []
        let input_weight_gradient = []
    
        if(is_hidden){
            loss_out_gradient = y
        }
        else{
            for(let i = 0; i < this.neuron_size; i++){
                loss_out_gradient.push(this.loss_function.partial_derivative(this.output[i], y[i]))
            }
        }
        
        for(let i = 0; i < this.neuron_size; i++){
            derivative_activation.push(this.activation.partial_derivative(this.output[i]))
        }
    
        for(let i = 0; i < this.neuron_size; i++){
            input_weight_gradient.push(this.input)
        }
    
        let old_weight: number[][]= [...this.weight]
        for(let i = 0; i < this.neuron_size; i++){
            for(let j = 0; j < this.input_size; j++){
                let loss_weight = loss_out_gradient[i] * derivative_activation[i] * input_weight_gradient[i][j]
                this.weight[i][j] = this.weight[i][j] - (learning_rate * loss_weight)
            }
        }
    
        for(let i = 0; i < this.neuron_size; i++){
            let loss_bias = loss_out_gradient[i] * derivative_activation[i] * 1
            this.bias[i] = this.bias[i] - (learning_rate * loss_bias)
        }
    
        let sum_loss_out_gradient = 0
        let sum_derivative_activation = 0
        let sum_input = []
        let sum_old_weight = []
    
        for(let i = 0; i < this.neuron_size; i++){
            sum_loss_out_gradient += loss_out_gradient[i]
        }
    
        for(let i = 0; i < this.neuron_size; i++){
            sum_derivative_activation += derivative_activation[i]
        }
    
        for(let i = 0; i < this.input_size; i++){
            let sum_input_partial = 0
            for(let j = 0; j < this.neuron_size; j++){
                sum_input_partial += input_weight_gradient[j][i]
            }
            sum_input.push(sum_input_partial)
        }
    
        for(let i = 0; i < this.input_size; i++){
            let sum_old_weight_partial = 0
            for(let j = 0; j < this.neuron_size; j++){
                sum_old_weight_partial += old_weight[j][i]
            }
            sum_old_weight.push(sum_old_weight_partial)
        }
        
        let loss_out_forward = []
    
        for(let i = 0; i < this.input_size; i++){
            loss_out_forward.push(sum_loss_out_gradient * sum_derivative_activation * sum_input[i] * sum_old_weight[i])
        }
    
        return loss_out_forward
    }
}