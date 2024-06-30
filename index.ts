import { IrisSetosaDetect } from "./iris_setosa_detect";

const Model: IrisSetosaDetect = new IrisSetosaDetect()


let input = [ 7, 3.3, 1.4, 0.2 ]
console.log("Input : ")
console.log(input)
console.log("Output : ")
console.log(Model.predict(input))