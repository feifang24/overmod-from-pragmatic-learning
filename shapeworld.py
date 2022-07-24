"""
Generate shapeworld reference games
"""

from shapely.geometry import Point, box
from shapely import affinity
import numpy as np
from numpy import random
import random as pyrandom
from PIL import Image
import aggdraw
from enum import Enum
from tqdm import tqdm
import os
import multiprocessing as mp
from collections import namedtuple
from typing import Optional, Dict
import yaml
import json
from shutil import copyfile

DIM = 64
X_MIN, X_MAX = (8, 48)
ONE_QUARTER = (X_MAX - X_MIN) // 3
X_MIN_34, X_MAX_34 = (X_MIN + ONE_QUARTER, X_MAX - ONE_QUARTER)
BUFFER = 10
SIZE_MIN, SIZE_MAX = (3, 8)

TWOFIVEFIVE = np.float32(255)

SHAPES = ['circle', 'square', 'rectangle', 'ellipse']
COLORS = ['red', 'blue', 'green', 'yellow', 'white', 'gray']
DEFAULT_COLOR = 'purple'
VOCAB = SHAPES + COLORS + ['shape']
BRUSHES = {c: aggdraw.Brush(c) for c in COLORS + [DEFAULT_COLOR]}
PENS = {c: aggdraw.Pen(c) for c in COLORS + [DEFAULT_COLOR]}

MAX_PLACEMENT_ATTEMPTS = 5

TYPICALITY_MAP = {
}  # can be overwritten after script is run; bad style plz forgive me


class ConfigProps(Enum):
    SHAPE_1_COLOR = 0
    SHAPE_1_SHAPE = 1
    SHAPE_2_COLOR = 2
    SHAPE_2_SHAPE = 3
    RELATION_DIR = 4


class SpecificationType(Enum):
    SHAPE = 0
    COLOR = 1
    BOTH = 2
    EITHER = 3


def rand_size():
    return random.randint(SIZE_MIN, SIZE_MAX)


def rand_size_2():
    """Slightly bigger."""
    return random.randint(SIZE_MIN + 2, SIZE_MAX + 2)


def rand_pos():
    return random.randint(X_MIN, X_MAX)


def random_shape(shapes=SHAPES):
    return random.choice(shapes)


def random_color(colors=COLORS):
    return random.choice(colors)


class Shape:
    def __init__(self,
                 x=None,
                 y=None,
                 relation=None,
                 relation_dir=None,
                 color=None,
                 color_interior=True,
                 rotate=False):
        self.color = color  # should never be None in our implementation
        self.color_interior = color_interior
        self.rotate = rotate
        if x is not None or y is not None:
            assert x is not None and y is not None
            assert relation is None and relation_dir is None
            self.x = x
            self.y = y
        elif relation is None and relation_dir is None:
            # our case -- generate 'single'
            self.x = rand_pos()
            self.y = rand_pos()
        else:
            # Generate on 3/4 of image according to relation dir
            if relation == 0:
                # x matters - y is totally random
                self.y = rand_pos()
                if relation_dir == 0:
                    # Place right 3/4 of screen, so second shape
                    # can be placed LEFT
                    self.x = random.randint(X_MIN_34, X_MAX)
                else:
                    # Place left 3/4
                    self.x = random.randint(X_MIN, X_MAX_34)
            else:
                # y matters - x is totally random
                self.x = rand_pos()
                if relation_dir == 0:
                    # Place top 3/4 of screen, so second shape can be placed
                    # BELOW
                    # NOTE: Remember coords for y travel in opp dir
                    self.y = random.randint(X_MIN, X_MAX_34)
                else:
                    self.y = random.randint(X_MIN_34, X_MAX)
        self.init_shape()

    def draw(self, image):
        image.draw.polygon(self.coords, PENS[self.color])

    def intersects(self, oth):
        return self.shape.intersects(oth.shape)

    @property
    def interior_shape(self):
        pass

    def _gen_random_points(self, num_points=1):
        """
        Returns a list of N randomly generated points within a polygon. 
        """
        buffer_size = 0.5

        min_x, min_y, max_x, max_y = self.interior_shape.bounds
        points = []
        i = 0
        while len(points) < num_points:
            random_point = Point(random.randint(min_x, max_x),
                                 random.randint(min_y,
                                                max_y)).buffer(buffer_size)
            if random_point.within(self.interior_shape):
                points.append(random_point)
            i += 1
        return points

    def color_interior_points(self, image):
        points = self._gen_random_points()
        for p in points:
            coords = p.bounds
            image.draw.ellipse(coords, BRUSHES[self.color])


class Ellipse(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size()
        # Dy must be at least 1.6x dx, to remove ambiguity with circle
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            assert smaller > SIZE_MIN, ("{} {}".format(smaller, self.dx))
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = Point(self.x, self.y).buffer(1)
        shape = affinity.scale(shape, self.dx, self.dy)
        # Rotation
        if self.rotate:
            shape = affinity.rotate(shape, random.randint(360))
        self.shape = shape

        self.coords = [int(x) for x in self.shape.bounds]
        # self.coords = np.round(np.array(self.shape.boundary).astype(np.int))
        #  print(len(np.array(self.shape.convex_hull)))
        #  print(len(np.array(self.shape.convex_hull.boundary)))
        #  print(len(np.array(self.shape.exterior)))
        # self.coords = np.unique(self.coords, axis=0).flatten()

    @property
    def interior_shape(self):
        return affinity.scale(
            Point(self.x, self.y).buffer(1), self.dx - 2, self.dy - 2)

    def draw(self, image):
        if self.color_interior:
            image.draw.ellipse(self.coords, BRUSHES[DEFAULT_COLOR])
            self.color_interior_points(image)
        else:
            image.draw.ellipse(self.coords, BRUSHES[self.color])


class Circle(Ellipse):
    def init_shape(self):
        self.r = rand_size()
        self.shape = Point(self.x, self.y).buffer(self.r)
        self.coords = [int(x) for x in self.shape.bounds]

    @property
    def interior_shape(self):
        return Point(self.x, self.y).buffer(self.r - 1)


class Rectangle(Shape):
    def init_shape(self, min_skew=1.5):
        self.dx = rand_size_2()
        bigger = int(self.dx * min_skew)
        if bigger >= SIZE_MAX:
            smaller = int(self.dx / min_skew)
            self.dy = random.randint(SIZE_MIN, smaller)
        else:
            self.dy = random.randint(bigger, SIZE_MAX)
        if random.random() < 0.5:
            # Switch dx, dy
            self.dx, self.dy = self.dy, self.dx

        shape = box(self.x, self.y, self.x + self.dx, self.y + self.dy)
        # Rotation
        if self.rotate:
            shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    @property
    def interior_shape(self):
        return box(self.x + 0.5, self.y + 0.5, self.x + self.dx - 1,
                   self.y + self.dy - 1)

    def draw(self, image):
        if self.color_interior:
            image.draw.polygon(self.coords, BRUSHES[DEFAULT_COLOR],
                               PENS[DEFAULT_COLOR])
            self.color_interior_points(image)
        else:
            image.draw.polygon(self.coords, BRUSHES[self.color],
                               PENS[self.color])


class Square(Rectangle):
    def init_shape(self):
        self.size = rand_size_2()
        shape = box(self.x, self.y, self.x + self.size, self.y + self.size)
        # Rotation
        if self.rotate:
            shape = affinity.rotate(shape, random.randint(90))
        self.shape = shape

        # Get coords
        self.coords = np.round(
            np.array(self.shape.exterior.coords)[:-1].flatten()).astype(
                np.int).tolist()

    @property
    def interior_shape(self):
        return box(self.x + 0.5, self.y + 0.5, self.x + self.size - 1,
                   self.y + self.size - 1)


SHAPE_IMPLS = {
    'circle': Circle,
    'ellipse': Ellipse,
    'square': Square,
    'rectangle': Rectangle,
    # TODO: Triangle, semicircle
}


class I:
    def __init__(self):
        self.image = Image.new('RGB', (DIM, DIM))
        #  self.draw = ImageDraw.Draw(self.image)
        self.draw = aggdraw.Draw(self.image)

    def draw_shapes(self, shapes, flush=True):
        for shape in shapes:
            shape.draw(self)
        if flush:
            self.draw.flush()

    def show(self):
        self.image.show()
        #  self.image.resize((64, 64), Image.ANTIALIAS).show()

    def array(self):
        return np.array(self.image, dtype=np.uint8)

    def float_array(self):
        return np.divide(np.array(self.image), TWOFIVEFIVE)

    def save(self, path, filetype='PNG'):
        self.image.save(path, filetype)


def random_shape_from_spec(color=None, shape=None):
    if color is None:
        color = random_color()
    if shape is None:
        shape = random_shape()
    return (color, shape)


SingleConfig = namedtuple('SingleConfig', ['color', 'shape'])


def accept_config(config):
    accept = True
    if config.shape in TYPICALITY_MAP:
        typicality = TYPICALITY_MAP[config.shape]
        if config.color != typicality['typical_color']:
            accept = (random.random() < typicality['minority_accept_prob'])
    return accept


def random_config_single(color=None, shape=None):
    config = SingleConfig(*random_shape_from_spec(color=color, shape=shape))
    if not accept_config(config):
        config = random_config_single(shape=config.shape)
    return config


def new_color(existing_color):
    new_c = existing_color
    while new_c == existing_color:
        new_c = random.choice(COLORS)
    return new_c


def new_shape(existing_shape):
    new_s = existing_shape
    while new_s == existing_shape:
        new_s = random.choice(SHAPES)
    return new_s


def fmt_config(config):
    if isinstance(config, SingleConfig):
        return _fmt_config_single(config)
    else:
        raise NotImplementedError(type(config))


def _fmt_config_single(config):
    color, shape = config
    shape_txt = 'shape'
    color_txt = ''
    if shape is not None:
        shape_txt = shape
    if color is not None:
        color_txt = color + ' '
    return '{}{}'.format(color_txt, shape_txt)


def invalidate_single(config, part_to_invalidate):
    color, shape = config
    if part_to_invalidate == 0:  # invalidate color only
        combo = (new_color(color), shape)
    elif part_to_invalidate == 1:  # invalidate shape only
        combo = (color, new_shape(shape))
    elif part_to_invalidate == 2:  # invalidate both
        combo = (new_color(color), new_shape(shape))
    return combo


def generate_single(mp_args):
    random.seed()
    n_images, correct, i, data_type, context, color_interior, rotate = mp_args
    imgs = np.zeros((n_images, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n_images, ), dtype=np.uint8)
    target_config = random_config_single(
    )  # always (color, shape), no None allowed

    if data_type == 'concept':
        n_target = 2
        n_distract = 2
    else:  # reference, default
        n_target = 1
        n_distract = n_images  # Never run out of distractors
    idx = 0
    combos = []
    while idx < n_images:
        # decide label for current image
        if n_target > 0:
            label = 1
            n_target -= 1
        elif n_distract > 0:
            label = 0
            n_distract -= 1
        else:
            label = (random.random() < correct)

        # if context != None:
        target_color, target_shape = target_config
        if label == 1:
            shape_ = target_shape
            color_ = target_color
        if label == 0:
            if context == 'SHAPE':
                color_, shape_ = invalidate_single(target_config, 1)
            elif context == 'COLOR':
                color_, shape_ = invalidate_single(target_config, 0)
            elif context == 'BOTH':
                color_, shape_ = invalidate_single(
                    target_config, idx - 1
                )  # 2nd arg is 0 for 1st distractor and 1 for 2nd distractor
            elif context == 'EITHER':
                color_, shape_ = invalidate_single(target_config, 2)
        # else:
        # shape generalization - train
            """
            if shape_ == 'square':
                square = True
            else:
                square = False
            while square:
                shape_ = random_shape()
                if shape_ != 'square':
                    square = False"""
            # shape generalization - test
            """
            if label == 1:
                shape_ = 'square'"""

            # color generalization - train
            """
            if color_ == 'red':
                red = True
            else:
                red = False
            while red:
                color_ = random_color()
                if color_ != 'red':
                    red = False"""
            # color generalization - test
            """
            if label == 1:
                color_ = 'red'"""

            # combo generalization - train
            """
            if (color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square'):
                combo = True
            else:
                combo = False
            while combo:
                color_ = random_color()
                shape_ = random_shape()
                if not ((color_ == 'red' and shape_ == 'circle') or (color_ == 'blue' and shape_ == 'square') or (color_ == 'green' and shape_ == 'rectangle') or (color_ == 'yellow' and shape_ == 'ellipse') or (color_ == 'white' and shape_ == 'circle') or (color_ == 'gray' and shape_ == 'square')):
                    combo = False"""
            # combo generalization - test
            """
            if label == 1:
                combos = [('red','circle'),('blue','square'),('green','rectangle'),('yellow','ellipse'),('white','circle'),('gray','square')]
                combo = combos[np.random.randint(0,len(combos))]
                color_ = combo[0]
                shape_ = combo[1]"""

        if (color_, shape_) not in combos:
            # accept; otherwise start over

            combos.append((color_, shape_))
            shape = SHAPE_IMPLS[shape_](color=color_,
                                        color_interior=color_interior,
                                        rotate=rotate)
            # location is initialized via rand_pos: random.randint(X_MIN, X_MAX)
            # dimensions and rotations are defined in each subclass of Shape in init_shape

            # Create image and draw shape
            img = I()
            img.draw_shapes([shape])
            imgs[idx] = img.array()
            labels[idx] = label
            idx += 1

    specification_type = SpecificationType[context]

    # postprocess `target_config`
    if context == 'SHAPE':
        target_config = (None, target_config.shape)
    elif context == 'COLOR':
        target_config = (target_config.color, None)
    elif context == 'EITHER':
        target_config = pyrandom.choice([(None, target_config.shape),
                                         (target_config.color, None)])
    target_config = SingleConfig(*target_config)

    return imgs, labels, target_config, i, specification_type, combos


def generate(n,
             n_images,
             correct,
             data_type='concept',
             img_func=generate_single,
             float_type=False,
             n_cpu=None,
             pool=None,
             do_mp=True,
             verbose=False,
             context=None,
             for_vis=False,
             color_interior=False,
             rotate=False):
    if not do_mp and pool is not None:
        raise ValueError("Can't specify pool if do_mp=True")
    if do_mp:  # multiprocessing; ignore by default
        pool_was_none = False
        if pool is None:
            pool_was_none = True
            if n_cpu is None:
                n_cpu = mp.cpu_count()
            pool = mp.Pool(n_cpu)

    if data_type == 'concept':
        if n_images == 4:
            print(
                "Warning: n_images == 4, min targets/distractors both 2, no variance"
            )
        else:
            assert n_images > 4, "Too few n_images"
    elif data_type == 'reference':  # default
        assert n_images > 1, "Too few n_images"
    else:
        raise NotImplementedError("data_type = {}".format(data_type))

    all_imgs = np.zeros((n, n_images, 64, 64, 3), dtype=np.uint8)
    all_labels = np.zeros((n, n_images), dtype=np.uint8)
    targets_by_specification_type = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.int32)
    distractors_by_specification_type = np.zeros(
        (len(SpecificationType), len(COLORS), len(SHAPES)), dtype=np.int32)
    configs = []
    combos_per_example = []

    mp_args = [(n_images, correct, i, data_type,
                context.upper() if context is not None else SpecificationType(
                    random.randint(4)).name, color_interior, rotate)
               for i in range(n)]
    if do_mp:
        gen_iter = pool.imap(img_func, mp_args)
    else:
        gen_iter = map(img_func, mp_args)  # feed mp_args into img_func
    if verbose:
        gen_iter = tqdm(gen_iter, total=n)

    curr_index = 0
    if for_vis: visualization_data = []
    for output in gen_iter:  # refer to generate_single for expected output
        if output is not None:
            imgs, labels, config, _, specification_type, combos = output
            if for_vis:
                visualization_data.append(output)
            else:
                all_imgs[curr_index, ] = imgs
                all_labels[curr_index, ] = labels
                configs.append(config)
                combos_per_example.append(
                    [' '.join(combo) for combo in combos])
            for i, (color, shape) in enumerate(combos):
                if i == 0:
                    targets_by_specification_type[specification_type.value,
                                                  COLORS.index(color),
                                                  SHAPES.index(shape)] += 1
                else:
                    distractors_by_specification_type[specification_type.value,
                                                      COLORS.index(color),
                                                      SHAPES.index(shape)] += 1
            curr_index += 1
            if curr_index == n:
                break

    if for_vis:
        ret = {
            'viz_data': visualization_data,
            'targets_by_category': targets_by_specification_type,
            'distractors_by_category': distractors_by_specification_type
        }
    else:
        if float_type:
            all_imgs = np.divide(all_imgs, TWOFIVEFIVE)
            all_labels = all_labels.astype(np.float32)
        langs = np.array([fmt_config(c) for c in configs], dtype=np.unicode)
        ret = {
            'imgs': all_imgs,
            'labels': all_labels,
            'langs': langs,
            'all_referents': np.asarray(combos_per_example, dtype=np.unicode),
            'targets_by_category': targets_by_specification_type,
            'distractors_by_category': distractors_by_specification_type
        }

    if do_mp and pool_was_none:  # Remember to close the pool
        pool.close()
        pool.join()

    return ret


def save_images(img_dir, data):
    # Save to test directory
    for instance_idx, (instance, instance_labels, *rest) in enumerate(data):
        for world_idx, (world,
                        label) in enumerate(zip(instance, instance_labels)):
            Image.fromarray(world).save(
                os.path.join(img_dir,
                             '{}_{}.png'.format(instance_idx, world_idx)))

    index_fname = os.path.join(img_dir, 'index.html')
    with open(index_fname, 'w') as f:
        # Sorry for this code
        f.write('''
            <!DOCTYPE html>
            <html>
            <head>
            <title>Shapeworld Fast</title>
            <style>
            img {{
                padding: 10px;
            }}
            img.yes {{
                background-color: green;
            }}
            img.no {{
                background-color: red;
            }}
            </style>
            </head>
            <body>
            {}
            </body>
            </html>
            '''.format(''.join('<h1>{}</h1><p>{}</p>'.format(
            ' '.join(fmt_config(config)), ''.join(
                '<img src="{}_{}.png" class="{}">'.format(
                    instance_idx, world_idx, 'yes' if label else 'no')
                for world_idx, (
                    world,
                    label) in enumerate(zip(instance, instance_labels))))
                               for instance_idx, (instance, instance_labels,
                                                  config,
                                                  *rest) in enumerate(data))))
    # np.savez_compressed('test.npz', imgs=data.imgs, labels=data.labels)


def process_typicality_map(typicality_map: Dict[str, Dict[str, float]]):
    '''
    Maps shape to (a map from colors to prob values). Assume all prob values sum to <= 1
    Colors not mentioned are treated as uniform.
    '''
    for shape, typicality in typicality_map.items():
        TYPICALITY_MAP[shape] = {
            f'typical_{attr}': val
            for attr, val in typicality.items()
        }
        typical_prob = typicality['prob']
        TYPICALITY_MAP[shape]['minority_accept_prob'] = (1 - typical_prob) / (
            typical_prob * (len(COLORS) - 1))


IMG_FUNCS = {
    'single': generate_single,
}

DATAGEN_CONFIG_KEYS = [
    'context_type',
    'shape_color_typicality',
    'visualize',
    'name',
    'eval_only',
    'color_interior',
]

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Fast ShapeWorld',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n_examples',
                        type=int,
                        default=1000,
                        help='Number of examples')
    parser.add_argument('--n_images',
                        type=int,
                        default=3,
                        help='Images per example')
    parser.add_argument('--correct',
                        type=float,
                        default=0.5,
                        help='Avg correct proportion of images (concept only)')
    parser.add_argument('--no_mp',
                        action='store_true',
                        help='Don\'t use multiprocessing')
    parser.add_argument('--data_type',
                        choices=['concept', 'reference'],
                        default='reference',
                        help='What kind of data to generate')
    parser.add_argument('--img_type',
                        choices=list(IMG_FUNCS.keys()),
                        default='single',
                        help='What kind of images to generate')
    parser.add_argument(
        '--gen_config_fn',
        default='./datagen_config.yaml',
        type=str,
        help='Use the specified .yaml file as generation config')

    args = parser.parse_args()

    img_func = IMG_FUNCS[args.img_type]

    with open(args.gen_config_fn) as f:
        datagen_config = yaml.load(f, Loader=yaml.FullLoader)
    for key in datagen_config:
        if key not in DATAGEN_CONFIG_KEYS:
            raise ValueError(
                f"\'{key}\' is not a valid argument for datagen_config.")
    print(json.dumps(datagen_config, indent=4))

    context_type = datagen_config.get('context_type')

    if 'shape_color_typicality' in datagen_config:
        process_typicality_map(datagen_config['shape_color_typicality'])
        print(
            'Typical color for each shape, and acceptance prob of minority colors:'
        )
        print(json.dumps(TYPICALITY_MAP, indent=4))

    if datagen_config.get('visualize', False):
        # generate data purely for visualization
        vis_dir = os.path.join('viz_data', datagen_config['name'])
        if not os.path.isdir(vis_dir): os.makedirs(vis_dir)
        n_vis_examples = 100
        data = generate(n_vis_examples,
                        args.n_images,
                        args.correct,
                        verbose=True,
                        data_type=args.data_type,
                        img_func=img_func,
                        do_mp=not args.no_mp,
                        context=context_type,
                        for_vis=True,
                        color_interior=datagen_config.get(
                            'color_interior', False),
                        rotate=False)
        np.save(os.path.join(vis_dir, 'targets_by_category'),
                data['targets_by_category'])
        np.save(os.path.join(vis_dir, 'distractors_by_category'),
                data['distractors_by_category'])
        save_images(vis_dir, data['viz_data'])
    else:
        data_dir = os.path.join('data', datagen_config['name'])
        if not os.path.isdir(data_dir): os.makedirs(data_dir)

        # save dataset config
        with open(f'{data_dir}/metadata.yaml', 'w') as f:
            yaml.dump(datagen_config, f, default_flow_style=False)

        files = [f'{data_dir}/reference-1000-eval.npz']
        if datagen_config.get('eval_only', False):
            args.n_examples = 5000
        else:
            files.extend(
                [f'{data_dir}/reference-1000-{i}.npz' for i in range(0, 75)])
        for file in files:
            data = generate(args.n_examples,
                            args.n_images,
                            args.correct,
                            verbose=True,
                            data_type=args.data_type,
                            img_func=img_func,
                            do_mp=not args.no_mp,
                            context=context_type,
                            color_interior=datagen_config.get(
                                'color_interior', False),
                            rotate=False)
            np.savez_compressed(file, **data)
