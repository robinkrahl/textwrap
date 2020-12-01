use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

// The benchmarks here verify that the complexity grows as O(*n*)
// where *n* is the number of characters in the text to be wrapped.

use lipsum::lipsum_words_from_seed;

const LINE_LENGTH: usize = 60;

/// Generate a lorem ipsum text with the given number of characters.
fn lorem_ipsum(length: usize) -> String {
    // The average word length in the lorem ipsum text is somewhere
    // between 6 and 7. So we conservatively divide by 5 to have a
    // long enough text that we can truncate below.
    let mut text = lipsum_words_from_seed(length / 5, 42);
    text.truncate(length);
    text
}

pub fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("String lengths");
    for length in [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400].iter() {
        let text = lorem_ipsum(*length);
        let options = textwrap::Options::new(LINE_LENGTH);
        group.bench_with_input(BenchmarkId::new("fill", length), &text, |b, text| {
            b.iter(|| textwrap::fill(text, &options));
        });

        let options = textwrap::Options::new(LINE_LENGTH).balanced(false);
        group.bench_with_input(BenchmarkId::new("fill_ragged", length), &text, |b, text| {
            b.iter(|| textwrap::fill(text, &options));
        });

        let options: textwrap::Options =
            textwrap::Options::new(LINE_LENGTH).splitter(Box::new(textwrap::HyphenSplitter));
        group.bench_with_input(BenchmarkId::new("fill_boxed", length), &text, |b, text| {
            b.iter(|| textwrap::fill(text, &options));
        });

        group.bench_with_input(BenchmarkId::new("fill_usize", length), &text, |b, text| {
            b.iter(|| textwrap::fill(text, LINE_LENGTH));
        });

        group.bench_function(BenchmarkId::new("fill_inplace", length), |b| {
            b.iter_batched(
                || text.clone(),
                |mut text| textwrap::fill_inplace(&mut text, LINE_LENGTH),
                criterion::BatchSize::SmallInput,
            );
        });

        #[cfg(feature = "hyphenation")]
        {
            use hyphenation::{Language, Load, Standard};
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("benches")
                .join("la.standard.bincode");
            let dictionary = Standard::from_path(Language::Latin, &path).unwrap();
            let options = options.splitter(dictionary);
            group.bench_with_input(BenchmarkId::new("hyphenation", length), &text, |b, text| {
                b.iter(|| textwrap::fill(text, &options));
            });
        };
    }
    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
