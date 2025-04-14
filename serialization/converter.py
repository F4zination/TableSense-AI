import enum
import pathlib
import pandas as pd


class FileFormat(enum.Enum):
    """
    Enum class for file formats.
    """
    HTML = ".html"
    EXCEL = ".xlsx"
    CSV = ".csv"
    MARKDOWN = ".md"
    JSON = ".json"
    XML = ".xml"

    def equal(self, value: str) -> bool:
        if self.value == value:
            return True
        else:
            return False


class Converter:

    data: list[pd.DataFrame]  = []

    def __init__(self):
        pass

    def convert(self, path_to_file: pathlib.Path, output_format: FileFormat, save_to_file: bool)-> str:
        """
        Convert the given data to a different format.
        Returns the converted data as a string.
        The new file will be saved in the same directory as the original file.

        :param path_to_file: Path to the file to be converted
        :param output_format: The format to convert the file to
        :param save_to_file: Whether to save the converted file to disk
        :return The converted data as a string
        """
        # Check if the input file exists
        if not path_to_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {path_to_file}")

        # Check if the input file format is supported
        if path_to_file.suffix not in FileFormat:
            raise ValueError(f"Unsupported input file format: {path_to_file.suffix}")



        # Perform the conversion
        if path_to_file.suffix == FileFormat.CSV.value:
            self.data.append(pd.read_csv(path_to_file))
        elif path_to_file.suffix == FileFormat.JSON.value:
            self.data.append(pd.read_json(path_to_file))
        elif path_to_file.suffix == FileFormat.HTML.value:
            self.data = pd.read_html(path_to_file)
        elif path_to_file.suffix == FileFormat.MARKDOWN.value:
            self._read_markdown(path_to_file)
        elif path_to_file.suffix == FileFormat.XML.value:
            self.data.append(pd.read_xml(path_to_file))
        elif path_to_file.suffix == FileFormat.EXCEL.value:
            self.data.append(pd.read_excel(path_to_file))
        else:
            raise ValueError(f"Unsupported input file format: {path_to_file.suffix}")

        # return the converted data as a string

        output = ""
        if output_format.value == FileFormat.CSV.value:
            for df in self.data:
                output += df.to_csv()
        elif output_format.value == FileFormat.JSON.value:
            for df in self.data:
                output += df.to_json()
        elif output_format.value == FileFormat.HTML.value:
            for df in self.data:
                output += df.to_html()
        elif output_format.value == FileFormat.MARKDOWN.value:
            for df in self.data:
                output += df.to_markdown()
        elif output_format.value == FileFormat.XML.value:
            for df in self.data:
                output += df.to_xml()
        elif output_format.value == FileFormat.EXCEL.value:
            for df in self.data:
                output += df.to_excel()
        else:
            raise ValueError(f"Unsupported output file format: {output_format}")

        # Save the converted data to a file if requested
        if save_to_file:
            output_path = path_to_file.with_suffix(f".{output_format.value}")
            with open(output_path, "w") as f:
                f.write(output)
            print(f"Converted file saved to: {output_path}")

        return output





    def _read_markdown(self, path_to_file: pathlib.Path) -> None:
        """
        Read a markdown file and convert it to a DataFrame.
        """
        with open(path_to_file, "r") as f:
            lines = f.readlines()

        # Extract the header and data
        header = lines[0].strip().split("|")[1:-1]
        data = [line.strip().split("|")[1:-1] for line in lines[2:]]

        # Create a DataFrame
        self.data = [pd.DataFrame(data, columns=header)]


