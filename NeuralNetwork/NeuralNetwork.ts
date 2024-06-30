import type { Dataset } from "../dataset"
import type { Layer } from "./Layer/Layer"
import type { LossFunction } from "./LossFunction/LossFunction"

export interface NeuralNetwork{
    learning_rate: number
    epoch: number
    layer: Layer[]
    dataset: Dataset
    LossFunc: LossFunction

    setup: Function
    train: Function
    predict: Function
}