// Copyright 2020 The Complex Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"io"
	"math"
	"math/cmplx"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"
)

const (
	// Width of data
	Width = 4
	// Middle is the size of the middle layer
	Middle = 3
	// Eta is the learning rate
	Eta = .6
)

// Pair is a pair of training dtaa
type Pair struct {
	iris.Iris
	Input  [][]complex128
	In     []complex128
	Output []complex128
}

// Factorial computes the factorial of a number
func Factorial(n int) int {
	if n > 0 {
		return n * Factorial(n-1)
	}
	return 1
}

// Value is a value
type Value struct {
	Class  int
	Values []float64
}

var colors = [...]color.RGBA{
	{R: 0xff, G: 0x00, B: 0x00, A: 255},
	{R: 0x00, G: 0xff, B: 0x00, A: 255},
	{R: 0x00, G: 0x00, B: 0xff, A: 255},
}

var names = [...]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

func plotData(vals []Value, name string, width int) {
	length := len(vals)
	values := make([]float64, 0, width*length)
	for _, val := range vals {
		values = append(values, val.Values...)
	}
	data := mat.NewDense(length, width, values)
	rows, cols := data.Dims()

	var pc stat.PC
	ok := pc.PrincipalComponents(data, nil)
	if !ok {
		return
	}

	k := 2
	var projection mat.Dense
	projection.Mul(data, pc.VectorsTo(nil).Slice(0, cols, 0, k))

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "iris"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"
	p.Legend.Top = true

	for i := 0; i < 3; i++ {
		label := names[i]
		points := make(plotter.XYs, 0, rows)
		for j := 0; j < rows; j++ {
			if j/50 != i {
				continue
			}
			points = append(points, plotter.XY{X: projection.At(j, 0), Y: projection.At(j, 1)})
		}

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		scatter.GlyphStyle.Color = colors[i]
		p.Add(scatter)
		p.Legend.Add(fmt.Sprintf("%s", label), scatter)
	}

	err = p.Save(8*vg.Inch, 8*vg.Inch, name)
	if err != nil {
		panic(err)
	}
}

func printTable(out io.Writer, headers []string, rows [][]string) {
	sizes := make([]int, len(headers))
	for i, header := range headers {
		sizes[i] = len(header)
	}
	for _, row := range rows {
		for j, item := range row {
			if length := len(item); length > sizes[j] {
				sizes[j] = length
			}
		}
	}

	last := len(headers) - 1
	fmt.Fprintf(out, "| ")
	for i, header := range headers {
		fmt.Fprintf(out, "%s", header)
		spaces := sizes[i] - len(header)
		for spaces > 0 {
			fmt.Fprintf(out, " ")
			spaces--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n| ")
	for i, header := range headers {
		dashes := len(header)
		if sizes[i] > dashes {
			dashes = sizes[i]
		}
		for dashes > 0 {
			fmt.Fprintf(out, "-")
			dashes--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n")
	for _, row := range rows {
		fmt.Fprintf(out, "| ")
		last := len(row) - 1
		for i, entry := range row {
			spaces := sizes[i] - len(entry)
			fmt.Fprintf(out, "%s", entry)
			for spaces > 0 {
				fmt.Fprintf(out, " ")
				spaces--
			}
			fmt.Fprintf(out, " |")
			if i < last {
				fmt.Fprintf(out, " ")
			}
		}
		fmt.Fprintf(out, "\n")
	}
}

func main() {
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	items, n := datum.Fisher, Factorial(Width)
	length := /*n **/ len(items)
	pairs := make([]Pair, 0, len(items))
	for _, item := range items {
		permutations := make([][]complex128, 0, n)
		c, i, a := make([]int, Width), 0, make([]complex128, Width)
		for j, measure := range item.Measures {
			a[j] = cmplx.Rect(measure, 0 /*float64(j)*math.Pi/2*/)
		}

		permutation := make([]complex128, Width)
		copy(permutation, a)
		permutations = append(permutations, permutation)

		for i < Width {
			if c[i] < i {
				if i&1 == 0 {
					a[0], a[i] = a[i], a[0]
				} else {
					a[c[i]], a[i] = a[i], a[c[i]]
				}

				permutation := make([]complex128, Width)
				copy(permutation, a)
				permutations = append(permutations, permutation)

				c[i]++
				i = 0
				continue
			}
			c[i] = 0
			i++
		}

		output := make([]complex128, 0, len(permutations[0]))
		for x, out := range permutations[0] {
			output = append(output, cmplx.Rect(cmplx.Abs(out), float64(x)*math.Pi/4))
		}
		pair := Pair{
			Iris:   item,
			Input:  permutations,
			In:     permutations[0],
			Output: output,
		}
		pairs = append(pairs, pair)
	}
	count := 0
	for _, pair := range pairs {
		count += len(pair.Input)
	}
	/*if count != length {
		panic("invalid length")
	}*/
	fmt.Println("pairs", count)

	rnd := rand.New(rand.NewSource(1))
	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rnd.Float64()+a, (b-a)*rnd.Float64()+a)
	}

	parameters := make([]*tc128.V, 0, 4)
	w0, b0 := tc128.NewV(Width, Width), tc128.NewV(Width)
	w1, b1 := tc128.NewV(Width, Width), tc128.NewV(Width)
	w2, b2 := tc128.NewV(Width, Width), tc128.NewV(Width)
	w3, b3 := tc128.NewV(Width, Width), tc128.NewV(Width)
	parameters = append(parameters,
		&w0, &b0, &w1, &b1,
		&w2, &b2, &w3, &b3)
	for _, p := range parameters {
		for i := 0; i < cap(p.X); i++ {
			p.X = append(p.X, random128(-1, 1))
		}
	}

	input, output := tc128.NewV(Width, length), tc128.NewV(Width, length)
	zeros := tc128.NewV(Width, length)
	for i := 0; i < cap(zeros.X); i++ {
		zeros.X = append(zeros.X, 0)
	}
	l2 := tc128.Sigmoid(tc128.Add(tc128.Mul(w2.Meta(), input.Meta()), b2.Meta()))
	l3 := tc128.Add(tc128.Mul(w3.Meta(), l2), b3.Meta())
	l4 := tc128.Complex(input.Meta(), l3)

	l0 := tc128.Sigmoid(tc128.Add(tc128.Mul(w0.Meta(), l4), b0.Meta()))
	l1 := tc128.Add(tc128.Mul(w1.Meta(), l0), b1.Meta())

	one := tc128.NewV(1)
	one.X = append(one.X, 1)
	variance := tc128.Sub(one.Meta(), tc128.Avg(tc128.Variance(tc128.T(l0))))

	cost := tc128.Avg(tc128.Quadratic(tc128.Complex(l1, zeros.Meta()), tc128.Complex(output.Meta(), zeros.Meta())))
	cost = tc128.Add(cost, variance)

	inputs, outputs := make([]complex128, 0, Width*length), make([]complex128, 0, Width*length)
	for _, pair := range pairs {
		inputs = append(inputs, pair.In...)
		outputs = append(outputs, pair.Output...)
	}
	input.Set(inputs)
	output.Set(outputs)

	iterations := 256
	pointsAbs, pointsPhase := make(plotter.XYs, 0, iterations), make(plotter.XYs, 0, iterations)
	for i := 0; i < iterations; i++ {
		for _, p := range parameters {
			p.Zero()
		}
		input.Zero()
		output.Zero()

		total := tc128.Gradient(cost).X[0]

		norm := float64(0)
		for _, p := range parameters {
			for _, d := range p.D {
				norm += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm = math.Sqrt(norm)
		if norm > 1 {
			scaling := 1 / norm
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d * complex(scaling, 0)
				}
			}
		} else {
			for _, p := range parameters {
				for l, d := range p.D {
					p.X[l] -= Eta * d
				}
			}
		}

		pointsAbs = append(pointsAbs, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		pointsPhase = append(pointsPhase, plotter.XY{X: float64(i), Y: float64(cmplx.Phase(total))})
	}

	plot := func(title, name string, points plotter.XYs) {
		p, err := plot.New()
		if err != nil {
			panic(err)
		}

		p.Title.Text = title
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, name)
		if err != nil {
			panic(err)
		}
	}
	plot("cost abs vs epochs", "cost_abs.png", pointsAbs)
	plot("cost phase vs epochs", "cost_phase.png", pointsPhase)

	readme, err := os.Create("README.md")
	if err != nil {
		panic(err)
	}
	defer readme.Close()

	{
		input := tc128.NewV(Width)
		l2 := tc128.Sigmoid(tc128.Add(tc128.Mul(w2.Meta(), input.Meta()), b2.Meta()))
		l3 := tc128.Add(tc128.Mul(w3.Meta(), l2), b3.Meta())
		l4 := tc128.Complex(input.Meta(), l3)

		l0 := tc128.Sigmoid(tc128.Add(tc128.Mul(w0.Meta(), l4), b0.Meta()))

		headers, rows := make([]string, 0, 1+2*Width), make([][]string, 0, length)
		headers = append(headers, "label")
		for i := 0; i < Width; i++ {
			headers = append(headers, fmt.Sprintf("abs %d", i))
			headers = append(headers, fmt.Sprintf("phase %d", i))
		}
		values, valuesAbs, valuesPhase := make([]Value, 0, 8), make([]Value, 0, 8), make([]Value, 0, 8)
		for i, pair := range pairs {
			inputs := make([]complex128, Width)
			for j, in := range pair.In {
				inputs[j] = in
			}
			input.Set(inputs)
			row := make([]string, 0, 1+2*Width)
			val := Value{
				Class: i / 50,
			}
			valAbs := Value{
				Class: i / 50,
			}
			valPhase := Value{
				Class: i / 50,
			}
			l0(func(a *tc128.V) bool {
				row = append(row, pair.Label)
				for _, value := range a.X {
					val.Values = append(val.Values, cmplx.Abs(value))
					val.Values = append(val.Values, cmplx.Phase(value))
					valAbs.Values = append(valAbs.Values, cmplx.Abs(value))
					valPhase.Values = append(valPhase.Values, cmplx.Phase(value))
					row = append(row, fmt.Sprintf("%.8f", cmplx.Abs(value)))
					row = append(row, fmt.Sprintf("%.8f", cmplx.Phase(value)))
				}
				return true
			})
			values = append(values, val)
			valuesAbs = append(valuesAbs, valAbs)
			valuesPhase = append(valuesPhase, valPhase)
			rows = append(rows, row)
		}
		printTable(readme, headers, rows)
		plotData(values, "iris.png", 2*Width)
		plotData(valuesAbs, "irisAbs.png", Width)
		plotData(valuesPhase, "irisPhase.png", Width)
	}
}
