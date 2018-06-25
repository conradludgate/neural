package main

import (
	"io/ioutil"
)

const (
	TRAINING_IMAGES  = 60000
	PIXELS_PER_IMAGE = 28 * 28
)

var (
	TrainI []byte
	TrainL []byte
	TestI  []byte
	TestL  []byte

	TrainIi int
	TrainLi int
	TestIi  int
	TestLi  int
)

func Load() {
	b, _ := ioutil.ReadFile("data/trainingdata")
	TrainI = b[16:]
	b, _ = ioutil.ReadFile("data/traininglabels")
	TrainL = b[8:]
	b, _ = ioutil.ReadFile("data/testingdata")
	TestI = b[16:]
	b, _ = ioutil.ReadFile("data/testinglabels")
	TestL = b[8:]
}

func process_image(b []byte) (f [784]float32) {
	for i, v := range b {
		f[i] = float32(v) / 255
	}

	return
}

func TrainImage(i int) [784]float32 {
	return process_image(TrainI[784*i : 784*i+784])
}

func TrainLabel(i int) int {
	return int(TrainL[i])
}

func TestImage(i int) [784]float32 {
	return process_image(TestI[784*i : 784*i+784])
}

func TestLabel(i int) int {
	return int(TestL[i])
}
