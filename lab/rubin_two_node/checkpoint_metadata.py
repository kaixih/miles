"""Metadata selection for legacy and current Megatron torch_dist checkpoints."""


def checkpoint_metadata(root, iteration):
    """Return ordered metadata paths; inspect pickle structure without executing it.

    GLOBAL/REDUCE/NEWOBJ become inert records. No pickle.loads, class imports,
    callable invocation, tensor reads or persistent-ID resolution are permitted.
    This validates embedded common-state metadata, not optimizer resume success.
    The function is self-contained so the operator can send its source over SSH.
    """
    import json
    import pickletools
    from pathlib import Path

    root = Path(root)
    directory = root / f'iter_{iteration:07d}'
    common = directory / 'common.pt'
    indexes = [p for p in (directory / '.metadata', directory / 'metadata.json') if p.is_file()]
    if not indexes:
        raise ValueError('Missing distributed checkpoint metadata')
    if common.is_file():
        return [common, root / 'rollout' / f'global_dataset_state_dict_{iteration}.pt', *indexes]
    metadata, config_path = directory / '.metadata', directory / 'metadata.json'
    if not metadata.is_file() or not config_path.is_file():
        raise ValueError('Missing common.pt without embedded-common format evidence')
    if not 0 < metadata.stat().st_size <= 32 * 1024**2 or not 0 < config_path.stat().st_size <= 1024**2:
        raise ValueError('Embedded metadata exceeds inspection limits')
    config = json.loads(config_path.read_text())
    if config.get('sharded_backend') != 'torch_dist' or config.get('sharded_backend_version') != 1:
        raise ValueError('Unrecognized embedded-common checkpoint format')

    class Node:
        def __init__(self, symbol, args):
            self.symbol, self.args, self.state = symbol, args, None

    stack, memo, mark = [], {}, object()

    def marked():
        pos = next((i for i in range(len(stack) - 1, -1, -1) if stack[i] is mark), None)
        if pos is None:
            raise ValueError('Invalid pickle mark')
        result = stack[pos + 1:]
        del stack[pos:]
        return result

    raw = metadata.read_bytes()
    result = None
    for count, (op, arg, position) in enumerate(pickletools.genops(raw)):
        if count > 2_000_000:
            raise ValueError('Too many metadata pickle operations')
        name = op.name
        if name in ('PROTO', 'FRAME'):
            continue
        if name == 'STOP':
            if len(stack) != 1 or position + 1 != len(raw):
                raise ValueError('Invalid metadata pickle termination')
            result = stack.pop()
            break
        if name == 'MARK': stack.append(mark)
        elif name == 'NONE': stack.append(None)
        elif name in ('NEWTRUE', 'NEWFALSE'): stack.append(name == 'NEWTRUE')
        elif name in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT',
                      'STRING', 'BINSTRING', 'SHORT_BINSTRING', 'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE',
                      'BINUNICODE8', 'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8'):
            stack.append(arg)
        elif name in ('EMPTY_DICT', 'EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_SET'):
            stack.append({'EMPTY_DICT': dict, 'EMPTY_LIST': list, 'EMPTY_TUPLE': tuple, 'EMPTY_SET': set}[name]())
        elif name in ('BINPUT', 'LONG_BINPUT', 'PUT', 'MEMOIZE'):
            memo[len(memo) if name == 'MEMOIZE' else int(arg)] = stack[-1]
        elif name in ('BINGET', 'LONG_BINGET', 'GET'): stack.append(memo[int(arg)])
        elif name == 'GLOBAL': stack.append(('GLOBAL', *arg.split(' ', 1)))
        elif name == 'STACK_GLOBAL':
            symbol, module = stack.pop(), stack.pop()
            stack.append(('GLOBAL', module, symbol))
        elif name in ('REDUCE', 'NEWOBJ'):
            args, symbol = stack.pop(), stack.pop()
            stack.append(Node(symbol, args))  # Never invoke a referenced callable/class.
        elif name == 'NEWOBJ_EX':
            kwargs, args, symbol = stack.pop(), stack.pop(), stack.pop()
            stack.append(Node(symbol, (args, kwargs)))
        elif name == 'BUILD':
            state = stack.pop()
            if not isinstance(stack[-1], Node):
                raise ValueError('BUILD on non-inert metadata object')
            stack[-1].state = state
        elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
            values = marked()
            stack.append(dict(zip(values[::2], values[1::2])) if name == 'DICT'
                         else {'TUPLE': tuple, 'LIST': list, 'FROZENSET': frozenset}[name](values))
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            size = int(name[-1]); values = tuple(stack[-size:]); del stack[-size:]; stack.append(values)
        elif name == 'APPEND':
            value = stack.pop(); stack[-1].append(value)
        elif name in ('APPENDS', 'ADDITEMS'):
            values = marked()
            stack[-1].extend(values) if name == 'APPENDS' else stack[-1].update(values)
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop(); stack[-1][key] = value
        elif name == 'SETITEMS':
            values = marked(); stack[-1].update(zip(values[::2], values[1::2]))
        elif name == 'POP': stack.pop()
        elif name == 'POP_MARK': marked()
        elif name == 'DUP': stack.append(stack[-1])
        else:
            raise ValueError('Unsupported metadata pickle operation: ' + name)
    expected = ('GLOBAL', 'torch.distributed.checkpoint.metadata', 'Metadata')
    if not isinstance(result, Node) or result.symbol != expected or not isinstance(result.state, dict):
        raise ValueError('Expected inert torch DCP Metadata record')
    entries = result.state.get('state_dict_metadata')
    key = 'common_state/shard_0_1'
    common_entry = entries.get(key) if isinstance(entries, dict) else None
    if (not isinstance(common_entry, Node) or common_entry.symbol !=
            ('GLOBAL', 'torch.distributed.checkpoint.metadata', 'BytesStorageMetadata')):
        raise ValueError('DCP metadata does not prove embedded common_state BytesStorageMetadata')
    return [root / 'rollout' / f'global_dataset_state_dict_{iteration}.pt', *indexes]
