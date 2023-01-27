use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

fn normalize_buf(buf: &mut Vec<Complex<f32>>) {
    let buf_len = buf.len();
    for x in &mut buf.iter_mut() {
        *x = *x / (buf_len as f32);
    }
}

fn main() {
    // create a test input vector
    let input_buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("Input buffer: {:?}\n", input_buf);
    // create a planner for the FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(input_buf.len());

    // convert the input to complex numbers
    let mut buf: Vec<Complex<f32>> = input_buf.iter().map(|x| Complex::new(*x, 0.0)).collect();
    println!("Input buffer (complex): {:?}\n", buf);

    // perform the FFT
    fft.process(&mut buf);
    println!("Post-FFT: {:?}\n", buf);

    // create a planner for the IDFT
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(buf.len());

    // perform the IDFT
    ifft.process(&mut buf);
    println!("Post-IDFT: {:?}\n", buf);

    // Normalize the output
    normalize_buf(&mut buf);

    // extract the real parts of the output
    let output_real: Vec<f32> = buf.iter().map(|x| x.re).collect();
    println!("Output buffer (real): {:?}\n", output_real);

    // check that the output is equal to the original input
    assert_eq!(input_buf, output_real);
}
