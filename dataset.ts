import { readFileSync } from "fs"
import path from "path"

export class Dataset{
    matrix: any[]
    label: string[]
    max: number[] = []
    min: number[] = []
    data = {
        x: [] as number[][],
        y: [] as number[][]
    }

    constructor(){
        this.matrix = []
        this.label = []
    }

    load(){
        let data: Buffer = readFileSync(path.join(__dirname) + "/dataset/iris.csv")
        let split: string[] = data.toString().split("/\n/g")[0].split(/\r?\n|\r|\n/g)
        split.map((v, i) => {
            let v_splitting = v.split(",")
            let mx: any[] = []
            if(i != 0){
                v_splitting.map((v: string, i) => {
                    if(i < v_splitting.length - 1){
                        mx.push(parseFloat(v))
                    }
                    else{
                        mx.push(<string>v.replaceAll("\"", ""))
                    }
                })
                this.matrix.push(mx)
            }
            else{
                v_splitting.map((v: string, i) => {
                    mx.push(<string>v.replaceAll("\"", ""))
                })
                this.label.push(<any>mx)
            }
        })
    }

    get_matrix(){
        return this.matrix
    }

    get_label(){
        return this.label
    }

    one_hot_encoder(categories: string[]){
        const uniqueCategories = Array.from(new Set(categories));
        const categoryIndexMap = new Map<string, number>();
        uniqueCategories.forEach((category, index) => {
            categoryIndexMap.set(category, index);
        });

        const oneHotEncoded: number[][] = [];

        categories.forEach(category => {
            const oneHotArray = new Array(uniqueCategories.length).fill(0);
            const index = categoryIndexMap.get(category);
            if (index !== undefined) {
                oneHotArray[index] = 1;
            }
            oneHotEncoded.push(oneHotArray);
        });
        return oneHotEncoded;
    }

    parse(){
        let y_label: string[] = []
        this.matrix.map((v, i) => {
            let row: number[] = []
            v.map((w: any, j: number) => {
                if(j < v.length - 1){
                    row.push(w)
                }
                else{
                    y_label.push(w)
                }
            })

            this.data.x.push(row)
        })
        this.data.y = this.one_hot_encoder(y_label)
    }

    min_max_norm(x: number, min: number, max: number){
        return ((x - min) / (max - min))
    }

    transform(x: number[]){
        let output: number[] = []
        if(x.length != this.min.length) throw Error("ERROR TRANSFORM DATASET")

        for(let i = 0; i < x.length; i++){
            output.push(this.min_max_norm(x[i], this.min[i], this.max[i]))
        }

        return output
    }

    set_min_max(data: number[][]){
        if(data.length == 0) throw new Error("Length Error")
        this.max = [...data[0]]
        this.min = [...data[0]]

        for(let i = 0; i < data.length; i++){
            for(let j = 0; j < data[i].length; j++){
                if(data[i][j] > this.max[j]){
                    this.max[j] = data[i][j]
                }
                if(data[i][j] < this.min[j]){
                    this.min[j] = data[i][j]
                }
            }
        }
    }

    async normalization(){
        this.set_min_max(this.data.x)
        for(let i = 0; i < this.data.x.length; i++){
            for(let j = 0; j < this.data.x[i].length; j++){
                this.data.x[i][j] = this.min_max_norm(this.data.x[i][j], this.min[j], this.max[j])
            }
        }
    }

    get(){
        this.normalization()
        return this.data
    }

    getBinaryLabel(index: number){
        let output: number[][] = []
        for(let i = 0; i < this.data.y.length; i++){
            output.push([this.data.y[i][index]])
        }

        return output
    }
}