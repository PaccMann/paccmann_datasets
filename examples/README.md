# pytoda - examples

Here we report some `pytoda` usage examples.
Example data can be downloaded [here](https://ibm.box.com/v/paccmann-pytoda-data).


## Create a SMILES language object

The example [smiles_language_creator.py](./smiles_language_creator.py) creates and export a [`SMILESLanguage`](../pytoda/smiles/smiles_language.py) object.

```console
(pytoda) $ python examples/smiles_language_creator.py -h
usage: smiles_language_creator.py [-h]
                                  [--add_start_and_stop | --no_start_and_stop]
                                  smi_path output_filepath

positional arguments:
  smi_path              path to a input .smi file
  output_filepath       path to a output .csv file

optional arguments:
  -h, --help            show this help message and exit
  --add_start_and_stop
  --no_start_and_stop
```

## Convert a .smi file into a .csv containing Morgan fingerprints

The example [smiles_to_fingerprints.py](./smiles_to_fingerprints.py) converts a .smi file into a .csv containing Morgan fingerprints with tunable radius and bits.

```console
(pytoda) $ python examples/smiles_to_fingerprints.py -h
usage: smiles_to_fingerprints.py [-h] [-r RADIUS] [-b BITS]
                                 smi_filepath output_filepath

positional arguments:
  smi_filepath          path to a input .smi file
  output_filepath       path to a output .csv file

optional arguments:
  -h, --help            show this help message and exit
  -r RADIUS, --radius RADIUS
                        radius for the fingerprints. Defaults to 2
  -b BITS, --bits BITS  number of bits composing the fingerprint. Defaults to
                        512
```

## Split .csv data

The example [split_csv_data.py](./split_csv_data.py) split `.csv` data in train and test with different configurations.

```console
(pytoda) $ python examples/split_csv_data.py -h
usage: split_csv_data.py [-h] -f FILEPATHS [FILEPATHS ...] -o OUTPUT_PATH -d
                         DATA_TYPE -m {random,file} [-s SEED]
                         [-t TEST_FRACTION] [-n NUMBER_OF_COLUMNS]
                         [-i INDEX_COL] [--separator SEPARATOR]
                         [--header HEADER]

optional arguments:
  -h, --help            show this help message and exit
  -f FILEPATHS [FILEPATHS ...], --filepaths FILEPATHS [FILEPATHS ...]
                        list of files to use to generate the splits
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to a input .smi file
  -d DATA_TYPE, --data_type DATA_TYPE
                        data type, used to generate the output file names
  -m {random,file}, --mode {random,file}
                        strategy used to split the data
  -s SEED, --seed SEED  seed used by the random generator. Defaults to 42
  -t TEST_FRACTION, --test_fraction TEST_FRACTION
                        portion of samples in testing data. Defaults to 0.1
  -n NUMBER_OF_COLUMNS, --number_of_columns NUMBER_OF_COLUMNS
                        number of columns used to generate a hash. Defaults to
                        12
  -i INDEX_COL, --index_col INDEX_COL
                        index column in the .csv flies. Defaults to 0
  --separator SEPARATOR
                        separators in the .csv files. Defaults to ","
  --header HEADER       header row in the .csv files. Defaults to 0
```

For more examples see other repositories in the [PaccMann organization](https://github.com/PaccMann).

