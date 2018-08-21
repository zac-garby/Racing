var canvas = document.getElementById("c")
var ctx = canvas.getContext("2d")

var population = []
var mutationRate = 0.1
var layers = [6, 4, 4, 2]

class Chromosome {
    constructor() {
        this.weights = []
        this.biases = []

        for (var i = 0; i < layers.length; i++) {
            var layerSize = layers[i]
            var nextLayerSize = layers[i+1]
            var layer = []

            for (var j = 0; j < layerSize; j++) {
                var node = []

                for (var k = 0; k < nextLayerSize; k++) {
                    node.push(Math.random())
                }
                layer.push(node)
            }
            this.weights.push(layer)
        }

        for (var i = 0; i < layers.length; i++) {
            var layer = []

            for (var j = 0; j < layers[i]; j++) {
                layer.push(Math.random())
            }
            this.biases.push(layer)
        }
    }

    mutate() {
        this.mapWeights(w => w + (Math.random()*mutationRate*2 - mutationRate))
        this.mapBiases(b => b + (Math.random()*mutationRate*2 - mutationRate))
    }

    mapWeights(fn) {
        for (var i = 0; i < this.weights.length; i++) {
            var layer = this.weights[i]
            for (var j = 0; j < layer.length; j++) {
                var node = layer[j]
                for (var k = 0; k < node.length; k++) {
                    var connection = node[k]
                    this.weights[i][j][k] = Math.min(Math.max(fn(connection), -1), 1)
                }
            }
        }
    }

    mapBiases(fn) {
        for (var i = 0; i < this.biases.length; i++) {
            var layer = this.biases[i]
            for (var j = 0; j < layer.length; j++) {
                this.biases[i][j] = Math.min(Math.max(fn(layer[j]), -1), 1)
            }
        }
    }

    evaluate(inputs) {
        var result = inputs
        
        // Iterate over the layers, except from the last one
        for (var i = 0; i < this.weights.length-1; i++) {
            var layerWeights = this.weights[i]
            var nextLayerBiases = this.biases[i+1]
            var out = []

            // Iterate over the biases of the next layer
            for (var j = 0; j < nextLayerBiases.length; j++) {
                var bias = nextLayerBiases[j]
                var weightedSum = 0

                for (var k = 0; k < result.length; k++) {
                    var weight = layerWeights[k][j]
                    var val = result[k]
                    weightedSum += weight * val
                }

                var z = weightedSum + bias

                out.push(1 / (1 + Math.exp(-z)))
            }

            result = out
        }

        return result
    }
}
