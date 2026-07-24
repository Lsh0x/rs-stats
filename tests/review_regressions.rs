// quick repro checks via test
#[test]
fn review_major_repros() {
    use rs_stats::Distribution;
    use rs_stats::distributions::gamma_distribution::Gamma;
    use rs_stats::distributions::student_t::StudentT;
    use rs_stats::utils::special_functions::noncentral_t_cdf;

    // MAJOR 1: tiny-shape Gamma quantiles (scipy: median = 4.4713e-31)
    let g = Gamma::new(0.01, 1.0).unwrap();
    let q50 = g.inverse_cdf(0.5).unwrap();
    let q10 = g.inverse_cdf(0.1).unwrap();
    let q1e6 = g.inverse_cdf(1e-6).unwrap();
    println!("q50={q50:e} q10={q10:e} q1e-6={q1e6:e}");
    assert!(q50 > 0.0 && q10 > 0.0 && q1e6 > 0.0);
    assert!(
        q1e6 < q10 && q10 < q50,
        "quantiles must be distinct & ordered"
    );
    let back = g.cdf(q50).unwrap();
    assert!((back - 0.5).abs() < 1e-9, "roundtrip cdf(q50) = {back}");

    // MAJOR 2: nct with huge nc must keep t-dependence
    let v = noncentral_t_cdf(50.0, 10.0, 40.0);
    println!("nct(50,10,40) = {v}");
    assert!((v - 0.7797).abs() < 0.01, "nct = {v}, scipy = 0.7797");
    assert!(noncentral_t_cdf(-50.0, 10.0, 40.0) < 1e-10);

    // MAJOR 3: StudentT p=0/1
    let t = StudentT::new(0.0, 1.0, 5.0).unwrap();
    assert_eq!(t.inverse_cdf(0.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(t.inverse_cdf(1.0).unwrap(), f64::INFINITY);
}
