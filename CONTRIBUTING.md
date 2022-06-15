# Contributing to blended-tiling

This project uses relatively simply linting and testing guidelines, as you'll see below.

## Linting


Linting is simple to perform.

```
pip install usort==1.0.2 black==22.3.0 flake8 mypy>=0.760 ufmt
```

Linting:

```
cd blended-tiling
black .
ufmt format .
cd ..
```

Checking:

```
cd blended-tiling
black --check --diff .
flake8 . --ignore=E203,W503 --max-line-length=88 --exclude build,dist
ufmt check .
mypy . --ignore-missing-imports --allow-redefinition
cd ..
```


## Testing

Tests can run like this:

```
pip install pytest pytest-cov
```

```
cd blended-tiling
pytest -ra --cov=. --cov-report term-missing
cd ..
```
