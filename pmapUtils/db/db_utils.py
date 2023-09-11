import json
import os
import warnings
from functools import partial
#from SciServer import Authentication

#import any_frame
import numpy as np
import pandas as pd
import random
import string

import pkg_resources
import sqlalchemy
import sqlalchemy.orm
import sqlalchemy.sql.expression
import urllib.parse
from typing import Optional
import getpass
import configparser

#from pmapUtils.db.resources import resources_root
#from pmapUtils.general.utils import get_environment, Environment

#######################################################################################
# FUNCTIONS TO DEFINE SQLALCHEMY CONNECTION ENGINE
#######################################################################################

def available_dbs():
    return {
        'VTE':'PatientSafetyQualityVTE_Projection',
        'WSP' : 'PatientSafetyQualityWSP_Projection',
        'MA':'PatientSafetyQualityMA_Projection',
        'camp' :  'CAMP_PMCoE_Projection'
    }


def get_connection_str(
    dbname,
    server = "ESMPMDBPR4.WIN.AD.JHU.EDU",
    **kwargs,
):
    """Returns a connection string to connect to a database

    Args:
        dbname (str): The name of the PMAP projection to connect to
        server (str): The name of the database server to connect to

    Returns:
        str: a database connection string
    """

   
    if 'user' not in kwargs:
        user = input('Username: ')
    else:
        user = kwargs.get("user")
   
    if 'passwd' not in kwargs:
        passwd = getpass.getpass('Password for ' + user + ': ')
    else:
        passwd = kwargs.get("passwd")
   
    user = "win\\" + user

    acceptable_servers = ["ESMPMDBPR4.WIN.AD.JHU.EDU"]
    
    if server == "ESMPMDBPR4.WIN.AD.JHU.EDU":
        connection_args = dict(
            DRIVER="FreeTDS",
            SERVER=server,
            PORT=1433,
            DATABASE=dbname,
            UID = user,
            PWD = passwd,
            TDS_VERSION = "8.0",
        )
         #create the connection string
        connection_args.update(kwargs)
        connection_args = "".join(f"{k}={v};" for k, v in connection_args.items())
        return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote(connection_args)}"
    else:
        raise KeyError(
            "Please specify a server type that current exists:  "
            + str(servers)
        )


def return_sql_engine(
    *args, **kwargs
):
    """Returns a SQL Server connection/engine object

    Returns:
        sqlalchemy.engine.base.Engine: a SQLAlchemy dB Engine
    """

    cnxn_string = get_connection_str(*args, **kwargs)
    engine = sqlalchemy.create_engine(cnxn_string)
    return engine


# define a default engine for subsequent functions to use
class EngineWrapper:
    def __init__(self, *args, engine=None, **kwargs):
        if engine is not None:
            connection = engine.connect()
        else:
            connection = None
        self._engine = engine
        self._connection = connection
        self._session = None
        self._engine_args = (args, kwargs)

    @property
    def engine(self):
        if self._engine is None:
            args, kwargs = self._engine_args
            self._engine = return_sql_engine(*args, **kwargs)
        return self._engine

    @property
    def connection(self):
        if self._connection is None:
            self._connection = self.engine.connect()
        return self._connection

    @property
    def session(self) -> sqlalchemy.orm.Session:
        if self._session is None:
            self._session = sqlalchemy.orm.Session(bind=self.connection)
        return self._session

    def inspect(self):
        return sqlalchemy.inspect(self.connection)

    def __getattr__(self, item):
        from .. import db

        try:
            func = getattr(db, item)
        except AttributeError:
            func = None
        if not callable(func) or isinstance(func, type):
            raise AttributeError(f"{self} has no attribute {item}")
        return partial(func, engine=self)


default_engine = EngineWrapper(dbname = available_dbs()["WSP"])
vte_default_engine = EngineWrapper(dbname = available_dbs()["VTE"])
wsp_default_engine = EngineWrapper(dbname = available_dbs()["WSP"])
ma_default_engine = EngineWrapper(dbname = available_dbs()["MA"])



########################################################################################################
# DATABASE UTILITIES

def glimpse(selectable,engine, size=6):
    """
    return the first `size` rows of a selectable
    """
    stmt = sqlalchemy.select(selectable).limit(size)
    return query_db(stmt,engine=engine)

def count(selectable,engine, by=None):
    if by is not None:
        stmt = (
            sqlalchemy.select(*[selectable.c[b] for b in by],
                              sqlalchemy.func.count().label("N"))
            .group_by(*[selectable.c[b] for b in by])
            .alias()
        )
        return query_db(stmt,engine=engine)
                      
        
    else:
        stmt = (
            sqlalchemy.select([sqlalchemy.func.count(selectable.columns[0]).label("N")])
            .select_from(selectable)
        )
        return query_db(stmt,engine=engine)["N"][0]
        

def random_name(length=16, self_seed=True):
    rng = random.Random() if self_seed else random
    return "".join(rng.choices(string.ascii_letters, k=length))


def query_db(
    query, temp_table=False, temp_table_name=None, engine=default_engine, **kwargs
):
    """Queries the database with a user-defined query string

    Args:
        query (str): a user-defined query
        temp_table (bool) : if True, the query will be converted to a temp table,
        (with optional name temp_table_name, or randomly generated name) on the db,
        rather than returning the results

    Returns:
        pandas.core.frame.DataFrame: a pandas dataframe with results from the query
    """
    if temp_table:
        return create_temp_table_query(query, temp_table_name, engine=engine)
    else:
        return pd.read_sql_query(sql=query, con=engine.connection, **kwargs)




def return_table(
    table,
    columns=None,
    max_rows=None,
    filter_condition=None,
    engine=default_engine,
):
    """Returns results of a single table query as defined by user with arguments

    Args:
        table (sqlalchemy.Table): database table
        columns ([str]): columns of table to include
        max_rows (int): maximum # of rows to include
        filter_condition (str): string representing 'WHERE' clause of a query
                            (Ex: "osler_id = '123456789'")
        engine (engineWrapper object)
    Returns:
        pandas.core.frame.DataFrame: a pandas dataframe with results from the query
    """
    if columns:
        query = sqlalchemy.select([table.columns[c] for c in columns]).select_from(
            table
        )
    else:
        query = table.select()

    if filter_condition is not None:
        query = query.where(sqlalchemy.text(filter_condition))

    if max_rows is not None:
        query = query.limit(max_rows)

    return pd.read_sql_query(sql=query, con=engine.connection)


def list_schemas(
    engine=default_engine,
):
    """Returns all database schemas

    Returns:
        [str]: list of schema names in database
    """
    insp = engine.inspect()
    return [
        schema
        for schema in insp.get_schema_names()
        if len(insp.get_table_names(schema=schema)) > 0
    ]


def list_tables(
    schema,
    engine=default_engine,
):
    """Returns all table names for a given schema

    Args:
        schema (str): name of schema to retrieve tables for

    Returns:
        list of str: list of table names for given schema
    """
    return engine.inspect().get_table_names(schema)


def list_columns(
    table,
    engine=default_engine,
):
    """Returns all columns for a given schema.table

    Args:
        table (sqlalchemy.Table): name of table to retrieve columns for

    Returns:
        list of str: list of column names for given schema.table
    """
    return [c.key for c in table.columns]


def load_table_definition(
    schema, table, engine=default_engine
) -> sqlalchemy.sql.expression.Selectable:
    
    try:
        return sqlalchemy.Table(
            table, sqlalchemy.MetaData(schema=schema), autoload_with=engine.connection
        )
    except sqlalchemy.exc.NoSuchTableError as ex:
        raise ValueError(f"{schema}.{table} could not be found") from ex

    

def get_table_dim(table, engine=default_engine):
    """Returns a dictionary specifying the dimensions (rows and columns) of a table

    Args:
        table (sqlalchemy.Table): database table

    Returns:
        dict: dictionary that specifies row/col dimensions of the table
    """
    return dict(
        rows=engine.session.query(table).count(),
        cols=len(table.columns),
    )


def get_keys(
    table,
    engine=default_engine,
):
    """Returns a dictionary that defines the PRIMARY KEY and FOREIGN KEY columns for a given table

    Args:
        table (sqlalchemy.Table): database table

    Returns:
        dict: dictionary that defines the PRIMARY KEY and FOREIGN KEY columns for a given table
    """
    return dict(
        PRIMARY=[c.name for c in table.primary_key.columns.values()],
        FOREIGN=[c.parent.name for c in table.foreign_keys],
    )


def get_column_info(
    schema,
    table,
    column,
    engine=default_engine,
):
    """Returns a dictionary of or prints information related to a specific table's column

    Args:
        schema (str): database schema
        table (str): database table
        column (str): database column
        verbose (bool): boolean specifies to print column definition or return dictionary of column info

    Returns:
         dict:  dictionary defining information for a given table's column
    """
    info_table = load_table_definition(
        schema, "DICTIONARY", engine
    )  # This is a table we generate...

    row = (
        sqlalchemy.select([info_table])
        .where(
            sqlalchemy.and_(
                info_table.columns["TABLE_NAME"] == table,
                info_table.columns["COLUMN_NAME"] == column,
            )
        )
        .distinct()
    )

    result = engine.connection.execute(row)

    k = result.keys()
    v = result.fetchone()
    if v is None:
        return None
    else:
        return {k[i]: v[i] for i in range(len(k))}




def make_temporary_table(*args, name=None, engine: EngineWrapper, **kwargs):
    name = name if name is not None else random_name()
    if engine.engine.dialect.name == "mssql":
        return sqlalchemy.Table(f"##{name.lower()}", *args, **kwargs)
    else:
        prefixes = kwargs.get("prefixes", [])
        kwargs["prefixes"] = prefixes + ["TEMPORARY"]
        return sqlalchemy.Table(name, *args, **kwargs)


# See pandas.io.SQLTable._sqlalchemy_type for reference
_dtype_to_sql_type_map = {
    "datetime": sqlalchemy.types.DateTime,  # Not going to bother supporting timestamps
    "datetime64": sqlalchemy.types.DateTime,
    "date": sqlalchemy.types.Date,
    "time": sqlalchemy.types.Time,
    "floating": {
        np.dtype("float32"): lambda: sqlalchemy.types.Float(precision=23),
        np.dtype("float64"): lambda: sqlalchemy.types.Float(precision=53),
        "default": sqlalchemy.types.Float,
    },
    "integer": {
        np.dtype("int32"): sqlalchemy.types.Integer,
        np.dtype("int64"): sqlalchemy.types.BigInteger,
        "default": sqlalchemy.types.Integer,
    },
    "boolean": sqlalchemy.types.Boolean,
}


def _get_column_sql_type(column: pd.Series):
    # A minimal port of pandas.io.SQLTable._sqlalchemy_type
    # import pandas.core.dtypes.inference.lib
    generic_dtype = pd.core.dtypes.inference.lib.infer_dtype(column)
    res = _dtype_to_sql_type_map.get(generic_dtype, sqlalchemy.types.Text)
    if isinstance(res, dict):
        try:
            res = res[column.dtype]
        except KeyError:
            res = res["default"]
    return res()


def make_temporary_table_from_pandas(
    df, *, pk_cols = None, engine: EngineWrapper, name=None, metadata=None, fixnan=True, rowchunks=None
) -> sqlalchemy.Table:
    
    if rowchunks is None:
        rowchunks = df.shape[0]
        
        
    def get_col_def(cname,col,pk_cols=None):
        if pk_cols is not None and cname in pk_cols:
            return sqlalchemy.Column(cname, _get_column_sql_type(col), primary_key=True)
        else:
            return sqlalchemy.Column(cname, _get_column_sql_type(col))
        
    def get_sqlalchemy_insertable(df,fixnan=True):
        k = list(df.to_dict("index").values())
        cols = df.select_dtypes("number").columns
        for i in range(len(k)):
            for j in cols:
                if pd.isna(k[i][j]):
                    k[i][j] = sqlalchemy.null()
        return k
        
    col_defs = [get_col_def(c,df[c], pk_cols) for c in df]
    
    metadata = metadata if metadata is not None else sqlalchemy.MetaData()
    table = make_temporary_table(
        metadata,
        *col_defs,
        name=name,
        engine=engine,
    )
    table.create(bind=engine.connection)
    
    # get insertable values
    insertable_values = get_sqlalchemy_insertable(df,fixnan=fixnan)
    
    # insert the values
    while insertable_values:
        iv_chunk, insertable_values = insertable_values[:rowchunks], insertable_values[rowchunks:]
        engine.session.execute(table.insert().values(iv_chunk))
        
    return table


def create_temp_table_values(
    values,
    colname,
    coltype,
    temp_table_name=None,
    engine=default_engine,
):
    """
    Function takes a list of values and a colname and creates a temp table
    on the database, which can then be used in join

    Args:
        values (list): The list of values
        colname (string): The name that should be assigned to this column of values in the temp table
        coltype (sqlalchemy.types.TypeEngine|type): The SQL Data type that these values represent
        temp_table_name: The name of the temp table to be created; if this doesn't begin with
        #, it will be appended to the beginning of the name. If not provided, a random name will be generated
        engine: the sql/connection engine, defaults to default_engine in db instance

    Returns:
        temp_table_name: returns back the temp_table_name
    """
    # Define and create the table
    table = make_temporary_table(
        sqlalchemy.MetaData(bind=engine.connection),
        sqlalchemy.Column(colname, coltype),
        name=temp_table_name,
        engine=engine,
    )
    table.create()

    # Insert the data
    # TODO: There appears to be a bug (prob in FreeTDS 7.2) that causes this to segfault... is this fixed more recently?
    engine.connection.execute(table.insert(), [{colname: value} for value in values])
    # for value in values:  # This is going to be a slow backup option
    #     table.insert().values({colname: value}).execute()

    return table


def create_temp_table_query(
    query,
    temp_table_name=None,
    pk_cols=None,
    engine=default_engine,
):
    """Places the result of a submitted query into a local temporary table on the database
    and returns the name of the temp_table

    Args:
        query (sqlalchemy.sql.expression.Selectable): any SQL query that can be executed on the database
        temp_table_name (str): a proposed name for the temporary table; a name
        will be prefixed with a hash if not provided (i.e. my_temp_table will
        be converted to #my_temp_table)

    Returns:
        str: the (hash-prefixed) name of the temporary table
    """
    # For now, need to enforce isinstance/selectable, and query.columns must not be empty
    if not isinstance(query, sqlalchemy.sql.expression.Selectable):
        raise TypeError(
            r"'query' argument must be a sqlalchemy.sql.expression.Selectable object".format()
        )
    # columns = query.subquery().columns
    columns = query.columns
    if len(columns) == 0:
        raise TypeError(
            r"'query' argument is Selectable, but columns collection is empty".format()
        )
        
    def get_col_def(c,pk_cols=None):
        if pk_cols is not None and c.name in pk_cols:
            return sqlalchemy.Column(c.key, c.type, primary_key=True)
        else:
            return sqlalchemy.Column(c.key,c.type)
        
    col_defs = [get_col_def(c, pk_cols) for c in columns]

    table = make_temporary_table(
        sqlalchemy.MetaData(bind=engine.connection),
        *col_defs,
        name=temp_table_name,
        engine=engine,
    )
    table.create()

    table.insert().from_select([c.key for c in columns], query).execute()

    return table


def gen_temp_indices(
    max_index,
    size=1000,
    seed=None,
    engine=default_engine,
):
    """Creates a temp table of random indices based on size and max_index

    Args:
        max_index (int): max index of indices to sample from
        size (int): size of table to return
        seed (int): random number generator seed

    Returns:
        tuple (str,int): tuple containing temp table name and count of rows
    """
    # if the size requested exceeds or meets max_index, just set the
    #  the indices to be series from 1 to max_index
    if size >= max_index:
        return_size = max_index
        rand_set = list(range(1, max_index + 1))
    else:
        # else return a k=size sample of the possible indices, without replacement
        return_size = size
        random.seed(seed)
        rand_set = random.sample(population=range(1, max_index + 1), k=size)

    return (
        create_temp_table_values(
            list(rand_set), "ID", sqlalchemy.types.Integer(), engine=engine
        ),
        return_size,
    )


def convert_input_to_db_string(
    input, colname=None, coltype=None, engine=default_engine
):
    """Detects the datatype of the input, and converts to an appropriate string. For
    example:
     - if string/table: return string as is
     - if string/query: wrap in parentheses
     - if local list of values: convert to temp table on database, given colname
     - if local data frame of values: convert the colname column of the input frame to temp table

    Args:
        input (pd.DataFrame, list, or str): a pd.DataFrame containing column named with <colname>,
                                            a list of colname values, or a str representing
                                            a table/temp table in the database
        colname (string): a name of column that will represent the input values; required if
        input is a local frame or list of values
        coltype (sqlalchemy.types.TypeEngine|type): a valid SQL datatype; required if input is a local frame or list
        of values

    Returns:
        input_db_string: db string representation of the input
    """

    # First, if this is a dataframe, grab the <colname> only, and convert to list
    if isinstance(input, pd.DataFrame):
        if colname is None:
            raise ValueError(
                r"No column name given, but required if passing local frame"
            )
        if colname not in list(input.columns):
            raise ValueError(
                rf"Local pandas data frame passed, but no column named {colname} found"
            )
        # convert the <colname> column to list
        input = input[colname].tolist()

    # Now, check for input as a string or list
    if isinstance(input, sqlalchemy.sql.expression.Selectable):
        return create_temp_table_query(input, engine=engine)
    elif isinstance(input, list):
        # since the input is local list of values, use create_temp_table_values()
        if coltype is None or colname is None:
            raise ValueError(
                r"Must pass colname and coltype if passing a local frame or list of values"
            )
        return create_temp_table_values(
            input, colname=colname, coltype=coltype, engine=engine
        )
    else:
        raise TypeError(
            r"'input' argument must be a pandas dataframe, list, or a string representing a dB table".format()
        )
