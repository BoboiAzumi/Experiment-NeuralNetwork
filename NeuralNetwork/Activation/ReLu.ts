import type { Activation } from "./Activation";

export class ReLu implements Activation{
    func(x: number){
        if(x > 0){
            return x;
        }
        return 0;
    }
    derivative(x: number) {
        if(x > 0){
            return 1;
        }
        return 0;
    }
    partial_derivative(y: number){
        if(y > 0){
            return 1;
        }
        return 0
    };
}