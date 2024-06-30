import type { Activation } from "./Activation";

export class Sigmoid implements Activation{
    func(x: number){
        return (1 / (1 + Math.pow(Math.E, -x)))
    }
    derivative(x: number) {
        return (this.func(x) * (1 - this.func(x)))
    }
    partial_derivative(y: number){
        return (y * (1 - y))
    }
}