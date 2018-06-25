package main

import (
	. "./go-bind"

	"math/rand"
	"time"
)

type predictreq struct {
	data   [784]float32
	result chan float32
}

var PredictQueue chan predictreq = make(chan predictreq)

func Train() {
	rand.Seed(time.Now().Unix())
	Load()

	go func() {
		nn := NewNN()
		order := make([]int, 60000)

		for i, _ := range order {
			order[i] = i
		}

		for {
			for i := 0; i < 60000; i++ {
				nn.Train(0.1, TrainImage(order[i]),
					target(TrainLabel(order[i])))

				// Predicts shouldn't run in parallel with trainer
				// so run all processes after training
				loop := true
				for loop {
					select {
					case pr := <-PredictQueue:
						m, o := output(nn.Predict(pr.data))
						pr.result <- float32(o) + m
						close(pr.result)

					default:
						loop = false
					}
				}
			}

			rand.Shuffle(60000, func(i, j int) {
				order[i], order[j] = order[j], order[i]
			})
		}

	}()
}

func output(outputs [10]float32) (m float32, o int) {
	for i, v := range outputs {
		if v > m {
			o = i
			m = v
		}
	}
	return
}

func target(t int) (_t [10]float32) {
	_t[t] = 1.0
	return _t
}

func Predict(image [784]float32) (int, float32) {
	pr := predictreq{
		image,
		make(chan float32),
	}

	PredictQueue <- pr
	v := <-pr.result
	i := int(v)
	return i, v - float32(i)
}
