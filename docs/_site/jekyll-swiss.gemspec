# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "jekyll-swiss"
  spec.version       = "1.0.0"
  spec.authors       = ["broccolini"]
  spec.email         = ["diana.mounter@gmail.com"]

  spec.summary       = %q{A bold typographic theme for Jekyll, inspired by Swiss design.}
  spec.homepage      = "http://broccolini.net/swiss"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(assets|_(includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end

  spec.add_development_dependency "jekyll", "~> 4.2"
  spec.add_development_dependency "bundler", "~> 2.7.2"
  spec.add_development_dependency "rake", "~> 12.0"
  spec.add_development_dependency "bigdecimal", "~> 3.0"
  spec.add_development_dependency "logger", "~> 1.4"
end
