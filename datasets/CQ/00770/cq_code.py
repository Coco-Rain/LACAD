import cadquery as cq

box1 = cq.Workplane("XY").box(10, 10, 1)
box2 = cq.Workplane("XY").transformed(offset=(5, 5, 0)).box(10, 10, 1)
result = box1.union(box2)
cq.exporters.export(result, 'GT.stl')