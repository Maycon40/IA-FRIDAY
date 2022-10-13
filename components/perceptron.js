export class RNPerceptron {
    constructor(config = {}){
        this._config = config;
        if(config.inputs){
            this.inputs = config.inputs;
        } else {
            this.inputs = [];
        }
        if(config.weights){
            this.weights = config.weights;
        } else {
            this.weights = [];
        }
        if(config.targets){
            this.targets = config.targets;
        } else {
            this.targets = [];
        }
        if(config.epochs){
            this.epochs = config.epochs;
        } else {
            this.epochs = 1;
        }
        if(config.activation){
            this.activation = config.activation;
        } else {
            this.activation = 'tanh';
        }
        if(config.hiddenLayers){
            this.hiddenLayers = config.hiddenLayers;
        } else {
            this.hiddenLayers = 2;
        }
        if(config.hiddenNodes){
            this.hiddenNodes = config.hiddenNodes;
        } else {
            this.hiddenNodes = 3;
        }
        if(config.bias){
            this.bias = config.bias;
        } else {
            this.bias = 1;
        }
        if(config.add){
            this.add = config.add;
        } else {
            this.add = [];
        }
    }

    train(fit = []){
        for(let i = 0; i < fit.length; i++){
            if(fit[i].input){
                this.inputs.push(fit[i].input);
            } else {
                this.inputs.push([0]);
            }
            if(fit[i].output){
                this.targets.push(fit[i].output);
            } else {
                this.targets.push([0]);
            }
        }

        //this.inputs = this.updateArray(this.inputs);
        //this.targets = this.updateArray(this.targets);

        for(let i = 0; i < this.inputs.length; i++){
            for(let j = 0; j < this.targets.length; j++){
                if((this.inputs[i][j] != undefined) && (this.targets[i][j] != undefined)){
                    this.feedforward(this.inputs[i], this.targets[i][j], this.epochs, this.hiddenLayers, this.hiddenNodes)
                }
            }

        }
    }

    sum(arr=[]){
        return arr.reduce((a, b) => a + b)
    }

    
    gradientDescent(n=0){
        return n * (1 - n)
    }

    updateArray(matrix = []){
        let resultMatrix = [];
        for(let i = 0; i < matrix.length; i++){
            let arr = matrix[i];
            let resultArray = [];
            for(let j = 0; j < arr.length; j++){
                let element = arr[j];
                if(element == 0) element = 0.01;
                if(element > 1){
                    let res = 0;
                    if(element.toString().trim().indexOf('.') > 0){
                        let temp = element.toString().trim();
                        let str = temp[0].toString().trim();
                        let len = str.length;
                        let div = '1'.padEnd(len + 1, '0');
                        res = element / div;
                        resultArray.push(res);
                    } else {
                        let str = element.toString().trim();
                        let len = str.length;
                        let div = '1'.padEnd(len + 1, '0');
                        res = element / div;
                        resultArray.push(res);
                    }
                } else {
                    resultArray.push(element);
                }
            }
            resultMatrix.push(resultArray)
        }

        return resultMatrix;
    }

    predict(inputs = []){
        //inputs = this.updateArray([inputs]);
        //inputs = inputs[0];
        let Outputs = [];
        // encontra a entrada do treino mais próxima da entrada da predição

        for(let i = 0; i < this.weights.length; i++){
            let input = this.weights[i].input;
            let diff = [];
            for(let j = 0; j < inputs.length; j++){
                diff.push(Math.abs(inputs[j] - input[j]));
            }
            let reduce = diff.reduce((a, b) => Number(a+''+b));
            this.add.push(reduce);
        }
        let search = inputs.reduce((a, b) => Number(a+''+b));
        let index = this.add.indexOf(search);

        for(let i = 0; i < this.targets[0].length; i++){
            //usa os pesos da entrada do treino mais próximo

            let matrixHidden = this.weights[index].weights;
            // sinapse das entradas para as ocultas

            let multiply = []
            for(let j = 0; j < inputs.length; j++){
                for(let x = 0; x < matrixHidden.length; x++){
                    for(let y = 0; y < matrixHidden[x].length; y++){
                        multiply.push(inputs[j] * matrixHidden[x][y]);
                    }
                }
            }

            let soma = this.sum(multiply);
            //função de ativação

            let output = 0;

            switch(this.activation){
            case 'tanh' : output = parseFloat(this.tanh(soma)).toFixed(4); break;
            case 'sigmoid' : output = parseFloat(this.sigmoid(soma)).toFixed(4); break;
            case 'relu' : output = parseFloat(this.relu(soma)).toFixed(4); break;
            case 'leakyRelu' : output = parseFloat(this.leakyRelu(soma)).toFixed(4); break;
            case 'binaryStep' : output = parseFloat(this.binaryStep(soma)).toFixed(4); break;
            default: output = parseFloat(this.tanh(soma)).toFixed(4); break;
            }
            //constroi o array de saida

            Outputs.push(Number(output));
        }

        return Outputs;
    }

    saveModel(path = './model.json'){
        fs.writeFileSync(path, JSON.stringify(this._config))
    }

    loadModel(path = './model.json'){
        const data = fs.readFileSync(path, 'utf8')
        const json = JSON.parse(data)
        this.X = json.input
        this.Y = json.output
    }

    feedforward(inputs = [], targets = 0, epochs = 1, hiddenLayers = 1, hiddenNodes = 2){
        // camada oculta

        let matrixHidden = [];
        for(let i = 0; i < hiddenLayers; i++){
            let arrHidden = [];
            for(let j = 0; j < hiddenNodes; j++){
                arrHidden.push(0);
            }
            matrixHidden.push(arrHidden)
        }

        //backpropagation

        let stop = false;
        let output = 0;
        if(targets != 0){
            for(let i = 1; i < epochs; i++){
                // sinapse das entradas para as ocultas

                let multiply = []
                for(let j = 0; j < inputs.length; j++){
                    for(let x = 0; x < matrixHidden.length; x++){
                        for(let y = 0; y < matrixHidden[x].length; y++){
                            multiply.push(inputs[j] * matrixHidden[x][y]);
                        }
                    }
                }
                let soma = this.sum(multiply);
                //função de ativação

                switch(this.activation){
                case 'tanh' : output = parseFloat(this.tanh(soma)).toFixed(4); break;
                case 'sigmoid' : output = parseFloat(this.sigmoid(soma)).toFixed(4); break;
                case 'relu' : output = parseFloat(this.relu(soma)).toFixed(4); break;
                case 'leakyRelu' : output = parseFloat(this.leakyRelu(soma)).toFixed(4); break;
                case 'binaryStep' : output = parseFloat(this.binaryStep(soma)).toFixed(4); break;
                default: output = parseFloat(this.tanh(soma)).toFixed(4); break;
                }
                //taxa de erro
    
                let error = parseFloat(Math.abs(targets - output)).toFixed(4);
                //corta o processamento quando encontrar um valor proximo da busca

                if((error <= 0.01) && (stop == false)){
                    this.weights.push({input: inputs, weights: matrixHidden});
                    i = epochs + 1;
                    stop = true;
                }
                //atualizar os pesos

                for(let j = 0; j < inputs.length; j++){
                    for(let x = 0; x < matrixHidden.length; x++){
                        for(let y = 0; y < matrixHidden[x].length; y++){
                            matrixHidden[x][y] += inputs[j] * this.gradientDescent(error);
                        }
                    }
                }
            }
        }
        //usa o bias se não encontrou a busca

        if(stop == false){
            if(output > targets){
                for(let x = 0; x < matrixHidden.length; x++){
                    for(let y = 0; y < matrixHidden[x].length; y++){
                        matrixHidden[x][y] -= this.bias;
                    }
                }
            } else if (output < targets){
                for(let x = 0; x < matrixHidden.length; x++){
                    for(let y = 0; y < matrixHidden[x].length; y++){
                        matrixHidden[x][y] += this.bias;
                    }
                }
            }

            this.weights.push({input: inputs, weights: matrixHidden});
        }
    }

    //  Tangente Hiperbólica  //

    tanh(n=0){
        return Math.sinh(n) / Math.cosh(n)
    }

    //  Função Sigmóide  //

    sigmoid(n=0){
        return 1 / (1 + Math.pow(Math.E, -n))
    }

    //  Função ReLU  //

    relu(n = 0){
        return Math.max(n, 0)
    }

    //  Função Leaky ReLU  //

    leakyRelu(n = 0){
        return Math.max(n, 0.01)
    }

    //  Função Binary Step  //

    binaryStep(n = 0){
        return (n >= 0) ? 1 : 0
    }
}