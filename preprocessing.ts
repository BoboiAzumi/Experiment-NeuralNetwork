import { Dataset } from "./dataset";

let dataset = new Dataset()

dataset.load()
dataset.parse()

console.log(dataset.get())