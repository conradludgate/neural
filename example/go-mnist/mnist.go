package main

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/nfnt/resize"
)

func main() {
	Train()

	http.HandleFunc("/", HandleHome)
	http.HandleFunc("/predict", HandlePredict)

	log.Fatal(http.ListenAndServe(":8080", nil))
}

func HandleHome(w http.ResponseWriter, r *http.Request) {
	b, _ := ioutil.ReadFile("index.html")
	w.Write(b)
}

func HandlePredict(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	i := image.NewRGBA(image.Rect(0, 0, 560, 560))
	i.Pix = b
	j := resize.Resize(28, 28, i, resize.NearestNeighbor)

	var d [784]float32

	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			c := color.GrayModel.Convert(j.At(x, y)).(color.Gray).Y
			d[x+y*28] = 1 - float32(c)/255
		}
	}

	o, m := Predict(d)
	w.Write([]byte(fmt.Sprintf("%d with %f%%", o, m*100)))
}
