var synaptic = require('synaptic')

var createRay = require("ray-aabb")

var canvas = document.getElementById("c")
var ctx = canvas.getContext("2d")

var layers = [6, 4, 4, 3, 2]

var mutationRate = 0.25
var population = 100
var carWidth = 5
var carHeight = 9
var carSpeed = 50
var carTurn = 30 * (Math.PI / 180)
var numFood = 30
var maxScanDist = 300

class Chromosome {
    constructor() {
        this.network = new synaptic.Architect.Perceptron(...layers)
    }

    copy() {
        var c = new Chromosome()
        c.network = synaptic.Network.fromJSON(this.network.toJSON())
        return c
    }

    mutate() {
        this.mapWeights(w => w + (Math.random()*mutationRate*2 - mutationRate))
        this.mapBiases(b => b + (Math.random()*mutationRate*2 - mutationRate))
    }

    mapWeights(fn) {
        var layers = [this.network.layers.input, ...this.network.layers.hidden, this.network.layers.output]
        layers.forEach(layer => {
            layer.list.forEach(neuron => {
                var proj = neuron.connections.projected

                for (var key in proj) {
                    if (proj.hasOwnProperty(key)) {
                        proj[key].weight = Math.min(Math.max(fn(proj[key].weight), 0), 1)
                    }
                }
            })
        })
        if (this.network.optimized) this.network.optimized.reset()
    }

    mapBiases(fn) {
        var layers = [this.network.layers.input, ...this.network.layers.hidden, this.network.layers.output]
        layers.forEach(layer => {
            layer.list.forEach(neuron => {
                neuron.bias = Math.min(Math.max(fn(neuron.bias), -1), 1)
            })
        })
        if (this.network.optimized) this.network.optimized.reset()
    }

    evaluate(inputs) {
        return this.network.activate(inputs)
    }
}

class Population {
    constructor(size, obstacles) {
        this.cars = []
        this.obstacles = obstacles
        this.foodPositions = []
        this.time = 0

        for (var i = 0; i < size; i++) {
            this.cars.push(new Car(
                Math.random()*150 + canvas.width/2 - 75,
                Math.random()*150 + canvas.height/2 - 75,
                Math.random() * 2 * Math.PI,
                new Chromosome(),
            ))
        }

        for (var i = 0; i < numFood; i++) {
            this.foodPositions.push([Math.random()*800+100, Math.random()*600+100])
        }
    }

    render(ctx) {
        this.obstacles.forEach(o => {
            var w = o[1][0] - o[0][0]
            var h = o[1][1] - o[0][1]
            ctx.fillStyle = "black"
            ctx.fillRect(o[0][0], o[0][1], w, h)
        })

        this.foodPositions.forEach(o => {
            ctx.fillStyle = "red"
            ctx.fillRect(o[0] - 2.5, o[1] - 2.5, 5, 5)
        })

        this.cars.forEach(c => c.render(ctx))
    }

    update(dt) {
        this.time += dt

        if (this.cars.every(c => c.dead) || this.time > 120) {
            this.generation()
            return
        }

        this.cars.forEach((function(c, i) {
            var bounds = c.bounds()
            if (this.collides(bounds, this.obstacles)) {
                this.kill(i)
                return
            }

            this.foodPositions.forEach((function(pos, i) {
                var x = pos[0]
                var y = pos[1]

                if (x > bounds[0][0] && y > bounds[0][1] && x < bounds[1][0] && y < bounds[1][1]) {
                    c.score++
                    this.foodPositions.splice(i, 1)
                    this.foodPositions.push([Math.random()*800+100, Math.random()*600+100])
                }
            }).bind(this))

            var inputs = this.networkInputs(i)
            c.update(dt, inputs)
        }).bind(this))
    }

    kill(index) {
        this.cars[index].dead = true
    }

    getSortedCars() {
        return this.cars.sort((a, b) => a.fitness() < b.fitness())
    }

    generation() {
        this.time = 0
        var sorted = this.getSortedCars()
        var best = sorted.slice(0, sorted.length/2)

        var mutated = []
        for (var car of best) {
            car.mutate().forEach(c => {
                mutated.push(new Car(
                    Math.random()*150 + canvas.width/2 - 75,
                    Math.random()*150 + canvas.height/2 - 75,
                    Math.random() * 2 * Math.PI,
                    c,
                ))
            })
        }
        
        this.cars = mutated
    }

    raycast(index, angle) {
        var car = this.cars[index]
        var pos = car.pos
        var totalAngle = angle + car.rotation
        var facing = [Math.cos(totalAngle), Math.sin(totalAngle)]
        var ray = createRay([pos.x, pos.y], facing)

        var min = Infinity
        for (var obstacle of this.obstacles) {
            var normal = [0, 0]
            var d = ray.intersects(obstacle, normal)
            if (!d) continue
            min = Math.min(min, d)
        }

        return min
    }

    scan(index) {
        return [-80, -30, 0, 30, 80]
            .map(deg => deg * (Math.PI / 180))
            .map(angle => this.raycast(index, angle))
            .map(dist => dist > maxScanDist ? 1 : dist / maxScanDist)
    }

    networkInputs(index) {
        var car = this.cars[index]
        var scan = this.scan(index)
        var dist = Infinity
        var closest = null

        for (var food of this.foodPositions) {
            var foodDist = Math.hypot(food[0] - car.pos.x, food[1] - car.pos.y)
            if (foodDist < dist) {
                dist = foodDist
                closest = food
            }
        }

        if (closest === null) return [...scan, 0]

        var angle = car.angleTo(...closest)

        return [...scan, angle]
    }

    collides(bounds, objects) {
        for (var obj of objects) {
            if (aabbaabb(bounds, obj)) return true
        }

        return false
    }
}

class Car {
    constructor(x, y, rot, chromosome) {
        this.pos = {x: x, y: y}
        this.vel = {x: 0, y: 0}
        this.acc = {x: 0, y: 0}
        this.rotation = rot
        this.rotationVel = 0
        this.rotationAcc = 0
        this.colour = `rgb(${Math.random()*255}, ${Math.random()*255}, ${Math.random()*255})`
        this.chromosome = chromosome
        this.score = 0
        this.counter = 0
        this.forwardAmount = 0
        this.turnAmount = 0
        this.dead = false
        this.lifetime = 0
    }

    fitness() {
        return this.score * this.lifetime
    }

    mutate() {
        var a = this.chromosome.copy()
        var b = this.chromosome.copy()
        a.mutate()
        b.mutate()
        return [a, b]
    }

    render(ctx) {
        ctx.save()
        ctx.translate(this.pos.x, this.pos.y)
        ctx.rotate(this.rotation)

        ctx.fillStyle = this.dead ? "rgba(0, 0, 0, 0.3)" : this.colour
        ctx.fillRect(-carWidth/2, -carHeight/2, carWidth, carHeight)

        ctx.restore()
    }

    update(dt, inputs) {
        if (this.dead) return

        this.lifetime += dt
        this.counter++

        if (this.counter >= 20) {
            var res = this.chromosome.evaluate(inputs)
            this.forwardAmount = res[0]
            this.turnAmount = res[1]
            this.counter = 0
        }

        this.forward(this.forwardAmount)
        this.turn(this.turnAmount)

        this.acc.x *= 0.9
        this.acc.y *= 0.9
        this.vel.x = (this.vel.x + this.acc.x * dt) * 0.95
        this.vel.y = (this.vel.y + this.acc.y * dt) * 0.95
        this.pos.x += this.vel.x * dt
        this.pos.y += this.vel.y * dt

        this.rotationAcc *= 0.9
        this.rotationVel = (this.rotationVel + this.rotationAcc * dt) * 0.95
        this.rotation += this.rotationVel * dt
        this.rotation %= Math.PI * 2
    }

    angleTo(x, y) {
        this.rotation %= Math.PI * 2
        var angle = Math.atan2(y, x)
        var diff = angle - this.rotation
        return normalizeAngle(diff)
    }

    forward(amount) {
        var speed = carSpeed * amount
        this.acc.x += Math.cos(this.rotation - Math.PI/2) * speed
        this.acc.y += Math.sin(this.rotation - Math.PI/2) * speed
    }

    turn(amount) {
        var angle = carTurn * 2 * (amount - 0.5)
        this.rotationAcc += angle
    }

    bounds() {
        var theta = this.rotation
        var sinTheta = Math.sin(theta)
        var cosTheta = Math.cos(theta)
        var x0 = this.pos.x
        var y0 = this.pos.y
        var w = carWidth
        var h = carHeight

        function transform(x, y) {
            return {
                x: x0 + (x-x0)*cosTheta - (y-y0)*sinTheta,
                y: y0 + (x-x0)*sinTheta + (y-y0)*cosTheta,
            }
        }

        var corners = [
            {x: x0-w/2, y: y0-h/2},
            {x: x0+w/2, y: y0-h/2},
            {x: x0-w/2, y: y0+h/2},
            {x: x0+w/2, y: y0+h/2},
        ]

        var transformed = corners.map(c => transform(c.x, c.y))
        var min = transformed.reduce((min, c) => {
            return {
                x: Math.min(min.x, c.x),
                y: Math.min(min.y, c.y),
            }
        }, {x: Infinity, y: Infinity})
        var max = transformed.reduce((max, c) => {
            return {
                x: Math.max(max.x, c.x),
                y: Math.max(max.y, c.y),
            }
        }, {x: -Infinity, y: -Infinity})

        return [[min.x, min.y], [max.x, max.y]]
    }
}

function normalizeAngle(rad) {
    return (rad % (Math.PI*2)) / (Math.PI*2) + 0.5
}

function unnormalizeAngle(n) {
    return (n - 0.5) * (Math.PI*2)
}

function aabbaabb(a, b) {
    var aw = Math.abs(a[1][0] - a[0][0])
    var bw = Math.abs(b[1][0] - b[0][0])
    var ah = Math.abs(a[1][1] - a[0][1])
    var bh = Math.abs(b[1][1] - b[0][1])
    var ac = [a[0][0] + aw/2, a[0][1] + ah/2]
    var bc = [b[0][0] + bw/2, b[0][1] + bh/2]
    var ar = [aw/2, ah/2]
    var br = [bw/2, bh/2]

    if (Math.abs(ac[0] - bc[0]) > (ar[0] + br[0])) return false
    if (Math.abs(ac[1] - bc[1]) > (ar[1] + br[1])) return false

    return true
}

var pop = new Population(population, [
    [[0, 0], [10, 900]],
    [[890, 0], [900, 900]],
    [[0, 0], [900, 10]],
    [[0, 890], [900, 900]],
    [[50, 50], [400, 200]],
    [[750, 500], [900, 900]],
    [[30, 600], [100, 750]],
    [[700, 40], [750, 100]],
    [[600, 300], [690, 310]],
])

var lastTime = performance.now()

function update() {
    var curTime = performance.now()
    var dt = (curTime - lastTime) / 1000
    lastTime = curTime

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    pop.update(dt)
    pop.render(ctx)

    requestAnimationFrame(update)
}

window.pop = pop

update()
