export class RedeNeural {
    train(config = {}){
        this._config = {}
        if(config.inputs){
        this.inputs = config.inputs
        } else {
        this.inputs = [0]
        }
        if(config.target){
        this.target = config.target
        } else {
        this.target = 0
        }
        if(config.epochs){
        this.epochs = config.epochs
        } else {
        this.epochs = 1
        }
        if(config.activation){
        this.activation = config.activation
        } else {
        this.activation = 'tanh'
        }
        this._config.inputs = this.inputs
        this._config.target = this.target
    }

    sum(arr=[]){
        return arr.reduce((a, b) => a + b)
    }

    
    gradientDescent(n=0){
        return n * (1 - n)
    }

    feedforward(){
        if(this.target<=0){
            this.target = 0.1
        } else if(this.target > 1){
            this.target = 1
        }

        let weights = []

        for(let i = 1; i < this.epochs; i++){
            let multiply = []
            for(let j = 0; j < this.inputs.length; j++){
            weights.push(Math.random())
            if(this.inputs[j] <= 0){
                this.inputs[j] = 0.1
            }
            multiply.push(this.inputs[j] * weights[j])
            }

            let soma = this.sum(multiply)
            let output = 0
            switch(this.activation){
            case 'tanh' : output = parseFloat(this.tanh(soma)).toFixed(4); break;
            case 'sigmoid' : output = parseFloat(this.sigmoid(soma)).toFixed(4); break;
            case 'relu' : output = parseFloat(this.relu(soma)).toFixed(4); break;
            case 'leakyRelu' : output = parseFloat(this.leakyRelu(soma)).toFixed(4); break;
            case 'binaryStep' : output = parseFloat(this.binaryStep(soma)).toFixed(4); break;
            default: output = parseFloat(this.tanh(soma)).toFixed(4); break;
            }

            let error = parseFloat(Math.abs(this.target - output)).toFixed(4)
            for(let j = 0; j < this.inputs.length; j++){
            if(this.inputs[j] <= 0){
                this.inputs[j] = 0.1
            }
            weights[j] += this.inputs[j] * this.gradientDescent(error)
            }
            let epoch = i.toString().padStart(7, '0')

            console.log(`época': ${epoch} - taxa de erro: ${error} - saída: ${output}`)
            return output
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