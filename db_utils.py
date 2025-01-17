import yaml
from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd

def load_credentials():
    ''' 
    Function loads database credentials from local yaml file and returns a dictionary.    
    '''

    with open('credentials.yaml', 'r') as file:
        db_credentials = yaml.safe_load(file)
    return db_credentials


class RDSDatabaseConnector:
    '''
    A class to create an instance of a RSD database connection.

    ...

    Attributes
    ----------
    credentials : dict
        dictionary of database credentials
    
    Methods
    -------
    engine_init():
        Initialises a SQLAlchemy engine from the credentials provided to RDSDatabaseConnector class.
    
    extract_data():
        Extracts data from the RDS database and returns it as a Pandas DataFrame.
    '''

    def __init__(self, credentials):
        '''
        Construct attributes for RDS Database Connector object.
        
        Parameters
        ----------
            credentials : dict
            dictionary of database credentials
        '''
        self.credentials = credentials
    
    def engine_init(self):
        '''
        Initialises a SQLAlchemy engine from the credentials provided to RDSDatabaseConnector class.
        '''

        DATABASE_TYPE = 'postgresql' #!!
        DBAPI = 'psycopg2' #!!
        HOST = self.credentials['RDS_HOST']
        USER = self.credentials['RDS_USER']
        PASSWORD = self.credentials['RDS_PASSWORD']
        DATABASE = self.credentials['RDS_DATABASE']
        PORT = self.credentials['RDS_PORT']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        
        return engine
    
    def extract_data(self, engine):
        '''
        Extracts data from the RDS database and returns it as a Pandas DataFrame. 
        The data is stored in a table called loan_payments.
        '''
        with engine.execution_options(isolation_level='AUTOCOMMIT').connect() as conn:
            
            # inspector = inspect(engine)
            # inspector.get_table_names()
            # table_names = inspector.get_table_names()
            # print("Tables in database:", table_names)
            loans_payments = pd.read_sql_table('loan_payments', engine)
            #loans_payments.head(1)
        return loans_payments
     
#load credentials
db_credentials = load_credentials()
# Create RDSDatabaseConnector instance
db = RDSDatabaseConnector(db_credentials)
# Initialize SQLAlchemy engine
SQLAlchemy_engine = db.engine_init()
# Extract data
loans_data = db.extract_data(SQLAlchemy_engine)
# Save data to CSV
loans_data.to_csv('loans_data.csv', index=False)
print("Data saved to loans_data.csv")