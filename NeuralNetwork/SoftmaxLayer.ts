/*
GAGAL !

import type { Layer } from "./Layer";

export class SoftmaxLayer implements Layer{
    sum: number = 0
    output: number[] = []
    input: number[] = []

    softmax(z: number){
        return Math.pow(Math.E, z) / this.sum
    }

    forward(data: number[]){
        let output = []
        this.input = data
        for(let i = 0; i < data.length; i++){
            this.sum += Math.pow(Math.E, data[i])
        }

        for(let i = 0; i < data.length; i++){
            output.push(this.softmax(data[i]))
        }

        this.output = output

        return output
    }
    backward(ayout: number[]){
        let output: number[] = []

        for(let i = 0; i < ayout.length; i++){
            let loss_out_gradient = - (ayout[i] / this.output[i])
            let gradient_input = loss_out_gradient * ayout[i] * (1 - this.output[i])
            output.push(gradient_input)
        }

        return output
    }
}*/