from pyspark.sql.types import *
import pyspark.sql.functions as F
import pyspark.sql
from typing import Iterable
import scipy.special as sps

def load_snps(spark, in_path):
    ''' Loads snp list for enrichment
    Params:
        in_path (file)
    Returns:
        spark.df
    '''
    # Load SNPs
    import_schema = StructType([
        StructField("study_id", StringType(), False),
        StructField("snp_id", StringType(), False),
        StructField("chrom", StringType(), False),
        StructField("pos", IntegerType(), False)
    ])
    snps = (
        spark.read.format('parquet').load(in_path)
    )
    return snps

def load_peaks(spark, in_path):
    ''' Loads peak coords and scores to wide df
    Params:
        in_path (file)
    Returns:
        spark.df
    '''
    # Load peaks. Can't specify schema due to variable column names.
    peaks_wide = (
        spark.read.csv(in_path, sep='\t', header=True, inferSchema=True)
        .repartitionByRange('chr', 'start', 'end')
    )

    return peaks_wide


def melt(df: pyspark.sql.DataFrame, 
        id_vars: Iterable[str], value_vars: Iterable[str], 
        var_name: str="variable", value_name: str="value") -> pyspark.sql.DataFrame:
    """Convert :class:`DataFrame` from wide to long format.
    Copied from: https://stackoverflow.com/a/41673644
    """

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = F.array(*(
        F.struct(F.lit(c).alias(var_name), F.col(c).alias(value_name)) 
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

    cols = id_vars + [
            F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)



def ndtr(x, mean, std):
    return .5 + .5*sps.erf((x - mean)/(std * 2**.5))
