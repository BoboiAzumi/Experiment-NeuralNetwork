import type { LossFunction } from "./LossFunction";

export class BinaryCrossEntropy implements LossFunction{
    loss(myout: number[], ayout: number[]){
        if(myout.length != ayout.length){
            throw new Error("Error length MSE")
        }
        let sum = 0;
        for(let i = 0; i < myout.length; i++){
            let a = ayout[i] * Math.log(myout[i] > 0 ? myout[i] : 0.00000000001)
            let b = (1 - ayout[i]) * Math.log(1 - (myout[i] < 1 ? myout[i] : 0.9999999999))
            sum += a + b
        }
        if(sum == 0){
            sum = -0
        }
    
        let bce = - ((1 / myout.length) * sum)
    
        return bce
    }
    
    partial_derivative(myout: number, ayout: number){
        return -((ayout / myout) - ((1 - ayout) / (1 - myout)))
    }
}