use criterion::{criterion_group, criterion_main};

// ── Poisson distribution ──────────────────────────────────────────────────────
mod poisson {
    use criterion::{BenchmarkId, Criterion, black_box};
    use rs_stats::Distribution;
    use rs_stats::distributions::poisson_distribution::Poisson;

    pub fn bench_pmf(c: &mut Criterion) {
        let mut group = c.benchmark_group("poisson_pmf");
        let p = Poisson::new(10.0).unwrap();
        for k in [5, 50, 200, 1000] {
            group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
                b.iter(|| p.pmf(black_box(k)));
            });
        }
        group.finish();
    }

    pub fn bench_cdf(c: &mut Criterion) {
        let mut group = c.benchmark_group("poisson_cdf");
        let p = Poisson::new(10.0).unwrap();
        for k in [5, 50, 200, 1000] {
            group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
                b.iter(|| p.cdf(black_box(k)));
            });
        }
        group.finish();
    }
}

// ── Binomial distribution ─────────────────────────────────────────────────────
mod binomial {
    use criterion::{BenchmarkId, Criterion, black_box};
    use rs_stats::Distribution;
    use rs_stats::distributions::binomial_distribution::Binomial;

    pub fn bench_cdf(c: &mut Criterion) {
        let mut group = c.benchmark_group("binomial_cdf");
        for (n, k) in [(20, 10), (100, 50), (500, 250)] {
            let dist = Binomial::new(n, 0.5).unwrap();
            group.bench_with_input(
                BenchmarkId::new("n_k", format!("{}_{}", n, k)),
                &k,
                |b, &k| {
                    b.iter(|| dist.cdf(black_box(k)));
                },
            );
        }
        group.finish();
    }

    pub fn bench_pmf(c: &mut Criterion) {
        let mut group = c.benchmark_group("binomial_pmf");
        for (n, k) in [(20, 10), (100, 50), (500, 250)] {
            let dist = Binomial::new(n, 0.5).unwrap();
            group.bench_with_input(
                BenchmarkId::new("n_k", format!("{}_{}", n, k)),
                &k,
                |b, &k| {
                    b.iter(|| dist.pmf(black_box(k)));
                },
            );
        }
        group.finish();
    }
}

// ── Hypothesis tests ──────────────────────────────────────────────────────────
mod hypothesis {
    use criterion::{BenchmarkId, Criterion, black_box};

    fn generate_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.7 + 1.5).collect()
    }

    pub fn bench_paired_t_test(c: &mut Criterion) {
        let mut group = c.benchmark_group("paired_t_test");
        for n in [10, 100, 1000, 10_000] {
            let data1 = generate_data(n);
            let data2: Vec<f64> = data1.iter().map(|x| x + 0.5).collect();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    rs_stats::hypothesis_tests::t_test::paired_t_test(
                        black_box(&data1),
                        black_box(&data2),
                    )
                });
            });
        }
        group.finish();
    }

    pub fn bench_one_sample_t_test(c: &mut Criterion) {
        let mut group = c.benchmark_group("one_sample_t_test");
        for n in [10, 100, 1000] {
            let data = generate_data(n);
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    rs_stats::hypothesis_tests::t_test::one_sample_t_test(black_box(&data), 5.0)
                });
            });
        }
        group.finish();
    }

    pub fn bench_anova(c: &mut Criterion) {
        let mut group = c.benchmark_group("anova");
        for n in [10, 100, 1000] {
            let g1 = generate_data(n);
            let g2: Vec<f64> = g1.iter().map(|x| x + 2.0).collect();
            let g3: Vec<f64> = g1.iter().map(|x| x - 1.0).collect();
            let groups_ref: Vec<&[f64]> = vec![&g1, &g2, &g3];
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| rs_stats::hypothesis_tests::anova::one_way_anova(black_box(&groups_ref)));
            });
        }
        group.finish();
    }
}

// ── Probability functions ─────────────────────────────────────────────────────
mod prob {
    use criterion::{BenchmarkId, Criterion, black_box};

    pub fn bench_erf(c: &mut Criterion) {
        let mut group = c.benchmark_group("erf");
        for x in [0.1, 1.0, 2.0, 3.0] {
            group.bench_with_input(BenchmarkId::from_parameter(x), &x, |b, &x| {
                b.iter(|| rs_stats::prob::erf(black_box(x)));
            });
        }
        group.finish();
    }
}

// ── Regression ────────────────────────────────────────────────────────────────
mod regression {
    use criterion::{BenchmarkId, Criterion, black_box};

    pub fn bench_linear_regression_fit(c: &mut Criterion) {
        let mut group = c.benchmark_group("linear_regression_fit");
        for n in [10, 100, 1000] {
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let y: Vec<f64> = x
                .iter()
                .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.01).sin())
                .collect();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    let mut model =
                        rs_stats::regression::linear_regression::LinearRegression::<f64>::new();
                    model.fit(black_box(&x), black_box(&y)).unwrap();
                });
            });
        }
        group.finish();
    }

    pub fn bench_linear_predict_many(c: &mut Criterion) {
        let mut group = c.benchmark_group("linear_predict_many");
        let x_train: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y_train: Vec<f64> = x_train.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let mut model = rs_stats::regression::linear_regression::LinearRegression::<f64>::new();
        model.fit(&x_train, &y_train).unwrap();

        for n in [100, 1000, 10_000] {
            let x_test: Vec<f64> = (0..n).map(|i| i as f64).collect();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| model.predict_many(black_box(&x_test)));
            });
        }
        group.finish();
    }

    pub fn bench_multiple_linear_regression(c: &mut Criterion) {
        let mut group = c.benchmark_group("mlr_fit");
        for n in [20, 100, 500] {
            let x: Vec<Vec<f64>> = (0..n)
                .map(|i| vec![i as f64, (i as f64) * 0.5, (i as f64) * 0.3])
                .collect();
            let y: Vec<f64> = x
                .iter()
                .map(|row| 2.0 * row[0] + 3.0 * row[1] - 1.5 * row[2] + 0.5)
                .collect();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    let mut model = rs_stats::regression::multiple_linear_regression::MultipleLinearRegression::<f64>::new();
                    model.fit(black_box(&x), black_box(&y)).unwrap();
                });
            });
        }
        group.finish();
    }
}

// ── Combinatorics ─────────────────────────────────────────────────────────────
mod combinatorics {
    use criterion::{BenchmarkId, Criterion, black_box};

    pub fn bench_factorial(c: &mut Criterion) {
        let mut group = c.benchmark_group("factorial");
        for n in [5, 10, 15, 20] {
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
                b.iter(|| rs_stats::utils::combinatorics::factorial(black_box(n)));
            });
        }
        group.finish();
    }

    pub fn bench_combination(c: &mut Criterion) {
        let mut group = c.benchmark_group("combination");
        for (n, k) in [(10, 5), (20, 10), (50, 25)] {
            group.bench_with_input(
                BenchmarkId::new("n_k", format!("{}_{}", n, k)),
                &(n, k),
                |b, &(n, k)| {
                    b.iter(|| {
                        rs_stats::utils::combinatorics::combination(black_box(n), black_box(k))
                    });
                },
            );
        }
        group.finish();
    }
}

// ── Distribution fitting ──────────────────────────────────────────────────────
mod fitting {
    use criterion::{BenchmarkId, Criterion, black_box};
    use rs_stats::Distribution;
    use rs_stats::distributions::student_t::StudentT;

    fn lognormal_ish(n: usize) -> Vec<f64> {
        // Deterministic right-skewed positive data (no rng dependency).
        (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                (2.0 * u).exp() * (1.0 + 0.1 * (i as f64).sin())
            })
            .collect()
    }

    pub fn bench_fit_all(c: &mut Criterion) {
        let mut group = c.benchmark_group("fit_all");
        group.sample_size(20);
        for n in [50, 1000, 10_000] {
            let data = lognormal_ish(n);
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| rs_stats::fit_all(black_box(&data)));
            });
        }
        group.finish();
    }

    pub fn bench_student_log_likelihood(c: &mut Criterion) {
        let mut group = c.benchmark_group("student_log_likelihood");
        let dist = StudentT::new(0.0, 1.0, 5.0).unwrap();
        for n in [1000, 10_000] {
            let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin() * 3.0).collect();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| dist.log_likelihood(black_box(&data)));
            });
        }
        group.finish();
    }
}

// ── Decision tree ─────────────────────────────────────────────────────────────
mod tree {
    use criterion::{BenchmarkId, Criterion, black_box};
    use rs_stats::regression::decision_tree::{DecisionTree, SplitCriterion, TreeType};

    pub fn bench_tree_fit(c: &mut Criterion) {
        let mut group = c.benchmark_group("tree_fit");
        group.sample_size(10);
        for n in [500_usize, 2000] {
            let features: Vec<Vec<i32>> = (0..n)
                .map(|i| (0..6).map(|f| ((i * 37 + f * 101) % 997) as i32).collect())
                .collect();
            let target_reg: Vec<i32> = (0..n).map(|i| ((i * 53) % 211) as i32).collect();
            group.bench_with_input(BenchmarkId::new("mse", n), &n, |b, _| {
                b.iter(|| {
                    let mut t = DecisionTree::<i32, f64>::new(
                        TreeType::Regression,
                        SplitCriterion::Mse,
                        8,
                        4,
                        2,
                    );
                    t.fit(black_box(&features), black_box(&target_reg)).unwrap();
                });
            });
            let target_cls: Vec<i32> = (0..n).map(|i| ((i * 7) % 5) as i32).collect();
            group.bench_with_input(BenchmarkId::new("gini", n), &n, |b, _| {
                b.iter(|| {
                    let mut t = DecisionTree::<i32, f64>::new(
                        TreeType::Classification,
                        SplitCriterion::Gini,
                        8,
                        4,
                        2,
                    );
                    t.fit(black_box(&features), black_box(&target_cls)).unwrap();
                });
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    poisson::bench_pmf,
    poisson::bench_cdf,
    fitting::bench_fit_all,
    fitting::bench_student_log_likelihood,
    tree::bench_tree_fit,
    binomial::bench_cdf,
    binomial::bench_pmf,
    hypothesis::bench_paired_t_test,
    hypothesis::bench_one_sample_t_test,
    hypothesis::bench_anova,
    prob::bench_erf,
    regression::bench_linear_regression_fit,
    regression::bench_linear_predict_many,
    regression::bench_multiple_linear_regression,
    combinatorics::bench_factorial,
    combinatorics::bench_combination,
);

criterion_main!(benches);
