import os.path
import pandas as pd
import sqlalchemy

from config import get_db_params
from sqlalchemy import create_engine
from sqlalchemy import MetaData, Table, String, Integer, Float, Column, Boolean, ForeignKey
from sqlalchemy import select, insert, text


def create_db(name: str = None):
    """
    Creates db according to database.ini. Returns True if db already exists
    :param name: name of database
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    params = get_db_params()
    user = params['user']
    password = params['password']
    if name is None:
        name = params['database']

    connection = psycopg2.connect(user=user, password=password)
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    cursor = connection.cursor()
    existed = False
    try:
        cursor.execute(f'create database {name}')
        print(f'Database "{name}" created.')
    except:
        existed = True
        print(f'Database "{name}" already exists.')
    cursor.close()
    connection.close()

    return existed


def get_engine() -> sqlalchemy.Engine:
    params = get_db_params()
    user = params['user']
    password = params['password']
    host = params['host']
    database = params['database']

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}/{database}")

    return engine


def get_meta() -> sqlalchemy.MetaData:
    metadata = MetaData()

    image = Table('Image', metadata,
                  Column('id', Integer(), primary_key=True, autoincrement=True),
                  Column('name', String(20), nullable=False, unique=True),
                  Column('label', String(20), nullable=False),
                  )

    sample = Table('Sample', metadata,
                   Column('id', Integer(), primary_key=True, autoincrement=True),
                   Column('name', String(50), nullable=False, unique=True),
                   )

    architecture = Table('Architecture', metadata,
                         Column('id', Integer(), primary_key=True, autoincrement=True),
                         Column('name', String(20), nullable=False, unique=True),
                         )

    hyperparams = Table('Hyperparams', metadata,
                        Column('id', Integer(), primary_key=True, autoincrement=True),
                        Column('optimizer', String(20), nullable=False),
                        Column('lr', Float(), nullable=False),
                        Column('wd', Float(), nullable=False),
                        Column('scheduler', String(20), nullable=False),
                        Column('sch_param', Float(), nullable=False),
                        )

    experiment = Table('Experiment', metadata,
                       Column('id', Integer(), primary_key=True, autoincrement=True),
                       Column('hyp_id', Integer(), ForeignKey(hyperparams.c.id)),
                       Column('arch_id', Integer(), ForeignKey(architecture.c.id)),
                       Column('model_name', String(50), nullable=False, unique=True),
                       Column('early_stopped', Boolean(), nullable=False),
                       Column('best_epoch', Integer(), nullable=False),
                       Column('last_epoch', Integer(), nullable=False),
                       Column('best_recall', Float(), nullable=False),
                       )

    info = Table('Info', metadata,
                 Column('id', Integer(), primary_key=True, autoincrement=True),
                 Column('exp_id', Integer(), ForeignKey(experiment.c.id)),
                 Column('train', Boolean(), nullable=False),
                 Column('epoch', Integer(), nullable=False),
                 Column('loss', Float(), nullable=False),
                 Column('accuracy', Float(), nullable=False),
                 Column('recall', Float(), nullable=False),
                 )

    sample_exp = Table('Sample_Experiment', metadata,
                       Column('sample_id', Integer(), ForeignKey(sample.c.id)),
                       Column('exp_id', Integer(), ForeignKey(experiment.c.id)),
                       Column('train', Boolean()),
                       )

    image_sample = Table('Image_Sample', metadata,
                         Column('image_id', Integer(), ForeignKey(image.c.id)),
                         Column('sample_id', Integer(), ForeignKey(sample.c.id)),
                         )

    return metadata


def add_images(df: pd.DataFrame, conn: sqlalchemy.Connection):
    df.index = df.index + 1
    df.to_sql('Image', con=conn, index=False, if_exists='append')


def add_sample(name: str, df: pd.DataFrame, meta: sqlalchemy.MetaData, conn: sqlalchemy.Connection):
    insert_query = meta.tables['Sample'].insert().values(name=name)
    conn.execute(insert_query)

    idxs = df.index + 1

    select_query = text('SELECT "Sample".id FROM "Sample" ORDER BY "Sample".id DESC')
    last_sample_id = conn.execute(select_query).fetchone()._asdict()['id']

    im_sam = {
        'image_id': idxs,
        'sample_id': [last_sample_id] * len(idxs)
    }
    im_sam = pd.DataFrame(im_sam)
    im_sam.to_sql('Image_Sample', con=conn, index=False, if_exists='append')


def add_arch(name: str, meta: sqlalchemy.MetaData, conn: sqlalchemy.Connection):
    arch = {
        'name': name,
    }

    insert_query = insert(meta.tables['Architecture']).values(arch)
    conn.execute(insert_query)


def save_results(results: dict):
    meta = get_meta()
    conn = get_engine().connect()

    conn.execute(insert(meta.tables['Hyperparams']).values(results['hyp']))
    select_hyp = text('SELECT "Hyperparams".id FROM "Hyperparams" ORDER BY "Hyperparams".id DESC')
    hyp_id = conn.execute(select_hyp).fetchone()._asdict()['id']

    arch_table = meta.tables['Architecture']
    select_arch = select(arch_table).where(arch_table.c.name.contains(results['arch_name']))
    arch_id = conn.execute(select_arch).fetchone()._asdict()['id']

    results['exp'].update({
        'hyp_id': hyp_id,
        'arch_id': arch_id,
        'model_name': f'Model_{hyp_id}',
    })
    conn.execute(insert(meta.tables['Experiment']).values(results['exp']))

    select_exp = text(f'SELECT "Experiment".id FROM "Experiment" ORDER BY "Experiment".id DESC')
    exp_id = conn.execute(select_exp).fetchone()._asdict()['id']

    results['info']['exp_id'] = [exp_id] * len(results['info'])
    results['info'].to_sql('Info', conn, if_exists='append', index=False)

    # sample_exp_train
    conn.execute(insert(meta.tables['Sample_Experiment']).values({
        'sample_id': 1,
        'exp_id': exp_id,
        'train': True,
    }))
    # sample_exp_test
    conn.execute(insert(meta.tables['Sample_Experiment']).values({
        'sample_id': 2,
        'exp_id': exp_id,
        'train': False,
    }))

    conn.commit()


if __name__ == "__main__":
    db_existed = create_db()

    engine = get_engine()
    conn = engine.connect()
    meta = get_meta()

    if not db_existed:
        meta.drop_all(conn)
        meta.create_all(conn)

        df = pd.read_csv(os.path.join('../data/datasets/current/complete datasets/diagnoses.csv'))
        add_images(df, conn)

        train = pd.read_csv(os.path.join('../data/datasets/current/complete datasets/test_raw.csv'))
        test = pd.read_csv(os.path.join('../data/datasets/current/complete datasets/test_raw.csv'))
        add_sample('Train (base images)', train, meta, conn)
        add_sample('Test (base images)', test, meta, conn)

        for name in 'ResNet50 DenseNet121 EfficientNetB4'.split():
            add_arch(name, meta, conn)
    else:
        x = conn.execute(text('SELECT * FROM "Architecture"')).fetchall()
        print(x)
        x = conn.execute(text('SELECT * FROM "Experiment"')).fetchall()
        print(x)

    conn.commit()
