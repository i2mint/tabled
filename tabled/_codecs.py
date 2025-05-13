"""
Codecs for various file formats to be used with tabled.

These codecs are only activated if the required dependencies are available.
Each codec section is wrapped in a try-except block to gracefully handle missing dependencies.
"""

from contextlib import suppress
from dol.util import read_from_bytes, written_bytes
from functools import partial
from tabled.wrappers import add_extension_codec

ignore_import_errors = suppress(ImportError, ModuleNotFoundError)

# MATLAB (.mat) files (requires: scipy)
with ignore_import_errors:
    import io
    import scipy.io
    import pandas as pd
    import numpy as np
    from typing import Union, Dict, Any

    def mat_bytes_to_dataframe(mat_bytes: Union[bytes, io.BytesIO]) -> pd.DataFrame:
        """
        Convert bytes of a .mat file to a pandas DataFrame.

        This function takes the binary content of a MATLAB .mat file,
        loads it using scipy.io, and converts the first variable found
        to a pandas DataFrame.

        Args:
            mat_bytes: Bytes object or BytesIO containing the .mat file data

        Returns:
            A pandas DataFrame containing the data from the .mat file

        """
        # Use BytesIO to create a file-like object from bytes
        if isinstance(mat_bytes, bytes):
            mat_file = io.BytesIO(mat_bytes)
        elif isinstance(mat_bytes, io.BytesIO):
            mat_file = mat_bytes
        else:
            raise ValueError("Input must be bytes or a BytesIO object")

        # Load the mat file
        mat_dict = scipy.io.loadmat(mat_file)

        # Filter out metadata entries (variables that start with '__')
        data_vars = {k: v for k, v in mat_dict.items() if not k.startswith("__")}

        if not data_vars:
            raise ValueError("No valid data variables found in the .mat file")

        # If there are multiple variables, we'll use the first one by default
        if len(data_vars) > 1:
            print(
                f"Multiple variables found in .mat file: {list(data_vars.keys())}. Using the first one."
            )

        # Get the first variable name and its data
        first_var_name = next(iter(data_vars))
        data = data_vars[first_var_name]

        # Convert to DataFrame, handling different data structures
        return _convert_to_dataframe(data, first_var_name)

    def _convert_to_dataframe(data: Any, var_name: str) -> pd.DataFrame:
        """
        Convert MATLAB data structure to pandas DataFrame.

        Args:
            data: Data loaded from the MATLAB file
            var_name: Variable name from the MATLAB file

        Returns:
            pandas DataFrame
        """
        # For struct arrays (MATLAB structs)
        if data.dtype.names is not None:
            # Convert structured array to dict of arrays
            return _convert_struct_to_dataframe(data)

        # Handle cell arrays (MATLAB cells)
        elif data.dtype == np.dtype("object"):
            return _convert_cell_to_dataframe(data, var_name)

        # Handle 2D numeric arrays
        elif len(data.shape) == 2 and np.issubdtype(data.dtype, np.number):
            return pd.DataFrame(data)

        # Handle 1D numeric arrays
        elif len(data.shape) == 1 and np.issubdtype(data.dtype, np.number):
            return pd.DataFrame(data, columns=[var_name])

        # Handle multidimensional arrays
        elif len(data.shape) > 2:
            return _convert_multidim_to_dataframe(data, var_name)

        # Last resort: try to manually extract values into a list of dictionaries
        else:
            try:
                # For 2D object arrays, try to extract values safely
                if len(data.shape) == 2:
                    rows = []
                    for i in range(data.shape[0]):
                        row = {}
                        for j in range(data.shape[1]):
                            # Try to get a scalar value
                            value = _extract_scalar(data[i, j])
                            row[f"col_{j}"] = value
                        rows.append(row)
                    return pd.DataFrame(rows)
            except Exception as e:
                print(f"Failed to extract data: {e}")

        # If all else fails
        raise ValueError(
            f"Could not convert MATLAB data to DataFrame. Data has shape {data.shape} "
            f"and dtype {data.dtype}. Consider using a more specific conversion function."
        )

    def _convert_struct_to_dataframe(data: np.ndarray) -> pd.DataFrame:
        """
        Convert a MATLAB struct array to a pandas DataFrame.

        Args:
            data: Structured numpy array from a MATLAB struct

        Returns:
            pandas DataFrame with columns for each field in the struct
        """
        # For a single struct, we create a single row DataFrame
        if data.shape == (1, 1):
            result = {}
            for field in data.dtype.names:
                # Extract the value and ensure it's a scalar
                result[field] = _extract_scalar(data[0, 0][field])
            return pd.DataFrame([result])

        # For multiple structs, create a DataFrame with one row per struct
        else:
            rows = []
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    row = {}
                    for field in data.dtype.names:
                        row[field] = _extract_scalar(data[i, j][field])
                    rows.append(row)
            return pd.DataFrame(rows)

    def _convert_cell_to_dataframe(data: np.ndarray, var_name: str) -> pd.DataFrame:
        """
        Convert a MATLAB cell array to a pandas DataFrame.

        Args:
            data: Object numpy array from a MATLAB cell array
            var_name: Variable name for generating column names

        Returns:
            pandas DataFrame representation of the cell array
        """
        # Try to determine if the first row contains column headers
        has_headers = False
        headers = []

        # If we have at least one row that might be headers
        if len(data.shape) == 2 and data.shape[0] > 0:
            try:
                # Check if first row elements are all strings
                if all(
                    isinstance(_extract_scalar(data[0, j]), str)
                    for j in range(data.shape[1])
                ):
                    has_headers = True
                    headers = [
                        _extract_scalar(data[0, j]) for j in range(data.shape[1])
                    ]
            except (IndexError, TypeError, AttributeError):
                has_headers = False

        # Process based on whether we have headers
        if has_headers:
            # Start from row 1 (skip headers)
            rows = []
            for i in range(1, data.shape[0]):
                row = {}
                for j, header in enumerate(headers):
                    if j < data.shape[1]:
                        row[header] = _extract_scalar(data[i, j])
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            # Convert all cells to a DataFrame
            rows = []
            for i in range(data.shape[0]):
                row = {}
                for j in range(data.shape[1]):
                    col_name = f"{var_name}_{j}" if data.shape[1] > 1 else var_name
                    row[col_name] = _extract_scalar(data[i, j])
                rows.append(row)
            return pd.DataFrame(rows)

    def _convert_multidim_to_dataframe(data: np.ndarray, var_name: str) -> pd.DataFrame:
        """
        Convert a multidimensional MATLAB array to a pandas DataFrame.

        Args:
            data: Multidimensional numpy array
            var_name: Variable name for generating column names

        Returns:
            pandas DataFrame representation of the multidimensional array
        """
        # Attempt to reshape to 2D
        reshaped = data.reshape(data.shape[0], -1)
        cols = [f"{var_name}_{i}" for i in range(reshaped.shape[1])]
        return pd.DataFrame(reshaped, columns=cols)

    def _extract_scalar(value: Any) -> Any:
        """
        Safely extract a scalar value from various MATLAB data types.

        Args:
            value: A value that might be a numpy array, matrix, or other MATLAB type

        Returns:
            A scalar Python value
        """
        # Handle arrays and matrices
        if isinstance(value, (np.ndarray, np.matrix)):
            # Empty arrays
            if value.size == 0:
                return None
            # Single value arrays
            elif value.size == 1:
                # Get the single value and convert if needed
                scalar = value.item() if hasattr(value, "item") else value[0]
                # Handle string arrays specifically
                if isinstance(scalar, np.ndarray) and scalar.dtype.type is np.str_:
                    return str(scalar)
                return scalar
            # For string arrays, join them
            elif value.dtype.type is np.str_:
                return " ".join(value.flatten())
            # For other arrays, return as list
            else:
                return value.tolist()
        # Handle other types
        else:
            return value

    def dataframe_to_mat_bytes(df: pd.DataFrame) -> bytes:
        """
        Convert a pandas DataFrame to MATLAB .mat file bytes.

        Args:
            df: pandas DataFrame to convert

        Returns:
            bytes: Binary content of a .mat file
        """
        buffer = io.BytesIO()
        # Convert DataFrame to dictionary where the key is 'data'
        data_dict = {"data": df.to_numpy()}

        # Add column names as a separate variable if they're not just numbers
        if not all(isinstance(col, (int, float)) for col in df.columns):
            col_names = np.array(df.columns, dtype=object)
            data_dict["column_names"] = col_names

        # Add index as a separate variable if it's not the default RangeIndex
        if (
            not isinstance(df.index, pd.RangeIndex)
            or df.index.start != 0
            or df.index.step != 1
        ):
            index_values = np.array(df.index, dtype=object)
            data_dict["index"] = index_values

        # Save to .mat format
        scipy.io.savemat(buffer, data_dict)
        return buffer.getvalue()

    # Register the .mat codec
    add_extension_codec(
        extension="mat",
        encoder=written_bytes(dataframe_to_mat_bytes),
        decoder=read_from_bytes(mat_bytes_to_dataframe),
    )


# SPSS Files (.sav) (requires: pyreadstat)
with ignore_import_errors:
    import pandas as pd
    import io
    import pyreadstat

    def sav_bytes_to_dataframe(sav_bytes: bytes) -> pd.DataFrame:
        """
        Convert bytes of a SPSS .sav file to a pandas DataFrame.

        Args:
            sav_bytes: Bytes object containing the SPSS file data

        Returns:
            A pandas DataFrame
        """
        buffer = io.BytesIO(sav_bytes)
        df, meta = pyreadstat.read_sav(buffer)
        # Store metadata as attributes of the DataFrame
        df.pyreadstat_metadata = meta
        return df

    # Register the SPSS codec
    add_extension_codec(
        extension="sav", decoder=read_from_bytes(sav_bytes_to_dataframe)
    )

# SAS Files (.sas7bdat) (requires: pyreadstat)
with ignore_import_errors:
    import pandas as pd
    import io
    import pyreadstat

    def sas_bytes_to_dataframe(sas_bytes: bytes) -> pd.DataFrame:
        """
        Convert bytes of a SAS .sas7bdat file to a pandas DataFrame.

        Args:
            sas_bytes: Bytes object containing the SAS file data

        Returns:
            A pandas DataFrame
        """
        buffer = io.BytesIO(sas_bytes)
        df, meta = pyreadstat.read_sas7bdat(buffer)
        # Store metadata as attributes of the DataFrame
        df.pyreadstat_metadata = meta
        return df

    # Register the SAS codec
    add_extension_codec(
        extension="sas7bdat", decoder=read_from_bytes(sas_bytes_to_dataframe)
    )

# STATA Files (.dta) with additional options (requires: pyreadstat)
with ignore_import_errors:
    import pandas as pd
    import io
    import pyreadstat

    def stata_bytes_to_dataframe(stata_bytes: bytes) -> pd.DataFrame:
        """
        Convert bytes of a STATA .dta file to a pandas DataFrame using pyreadstat.

        This provides more metadata than pandas' built-in reader.

        Args:
            stata_bytes: Bytes object containing the STATA file data

        Returns:
            A pandas DataFrame with metadata
        """
        buffer = io.BytesIO(stata_bytes)
        df, meta = pyreadstat.read_dta(buffer)
        # Store metadata as attributes of the DataFrame
        df.pyreadstat_metadata = meta
        return df

    # Register the enhanced STATA codec
    add_extension_codec(
        extension="dta_meta", decoder=read_from_bytes(stata_bytes_to_dataframe)
    )

# Avro files (.avro) (requires: fastavro)
with ignore_import_errors:
    import pandas as pd
    import io
    import fastavro

    def avro_bytes_to_dataframe(avro_bytes: bytes) -> pd.DataFrame:
        """
        Convert bytes of an Avro file to a pandas DataFrame.

        Args:
            avro_bytes: Bytes object containing the Avro file data

        Returns:
            A pandas DataFrame
        """
        buffer = io.BytesIO(avro_bytes)
        avro_reader = fastavro.reader(buffer)
        records = list(avro_reader)
        return pd.DataFrame.from_records(records)

    def dataframe_to_avro_bytes(df: pd.DataFrame, schema=None) -> bytes:
        """
        Convert a pandas DataFrame to Avro file bytes.

        Args:
            df: pandas DataFrame to convert
            schema: Optional Avro schema. If None, it will be inferred.

        Returns:
            bytes: Binary content of an Avro file
        """
        buffer = io.BytesIO()
        records = df.to_dict("records")

        if schema is None:
            # Infer schema from DataFrame
            schema = {
                "namespace": "dataframe.avro",
                "type": "record",
                "name": "DataFrame",
                "fields": [
                    {
                        "name": str(column),
                        "type": ["null", "string", "int", "float", "boolean"],
                    }
                    for column in df.columns
                ],
            }

        fastavro.writer(buffer, schema, records)
        return buffer.getvalue()

    # Register the Avro codec
    add_extension_codec(
        extension="avro",
        encoder=written_bytes(dataframe_to_avro_bytes),
        decoder=read_from_bytes(avro_bytes_to_dataframe),
    )

# Arrow files (.arrow) (requires: pyarrow)
with ignore_import_errors:
    import pandas as pd
    import io
    import pyarrow as pa
    import pyarrow.feather as feather

    def arrow_bytes_to_dataframe(arrow_bytes: bytes) -> pd.DataFrame:
        """
        Convert bytes of an Arrow file to a pandas DataFrame.

        Args:
            arrow_bytes: Bytes object containing the Arrow file data

        Returns:
            A pandas DataFrame
        """
        buffer = io.BytesIO(arrow_bytes)
        return feather.read_feather(buffer)

    def dataframe_to_arrow_bytes(df: pd.DataFrame) -> bytes:
        """
        Convert a pandas DataFrame to Arrow file bytes.

        Args:
            df: pandas DataFrame to convert

        Returns:
            bytes: Binary content of an Arrow file
        """
        buffer = io.BytesIO()
        feather.write_feather(df, buffer)
        return buffer.getvalue()

    # Register the Arrow codec
    add_extension_codec(
        extension="arrow",
        encoder=written_bytes(dataframe_to_arrow_bytes),
        decoder=read_from_bytes(arrow_bytes_to_dataframe),
    )
