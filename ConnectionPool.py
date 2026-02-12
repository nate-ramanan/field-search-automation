from configparser import ConfigParser
from psycopg2.pool import SimpleConnectionPool

file = r'c:\Users\owner\Documents\Gameplay_FieldAutomation_full_v3\Gameplay_FieldAutomation_full_v3\config.ini'
config = ConfigParser()
config.read(file)

print(config.sections())  # For debugging

hostname = config.get('database', 'hostname')
database = config.get('database', 'database')
username = config.get('database', 'username')
password = config.get('database', 'password')
port_id = config.get('database', 'port_id')

pool = SimpleConnectionPool(
    1, 20,
    user=username,
    password=password,
    host=hostname,
    port=int(port_id),
    database=database
)
print("Connection pool created successfully")

