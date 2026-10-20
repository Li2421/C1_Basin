from common import *
import ast
p=D/'blackbox_addendum_v1/continuing_rhr.py';s=p.read_text();tree=ast.parse(s);node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RightHand');body=ast.get_source_segment(s,node)
(HERE/'right_hand.py').write_text('"""Verbatim historical RightHand class; isolated safe imports."""\nfrom common import *\nfrom trigger import Trigger\nfrom poc.field_controller import Runtime\n\n'+body+'\n')
write(HERE/'RIGHT_HAND_CLASS_SOURCE.json',dict(source=str(p),source_sha256=sha(p),class_first_line=node.lineno,class_last_line=node.end_lineno,class_sha256=hashlib.sha256(body.encode()).hexdigest(),class_logic_unchanged=True))
