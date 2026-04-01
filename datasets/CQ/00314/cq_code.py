import cadquery as cq

box1 = cq.Workplane("XY").box(10, 10, 10)
box2 = cq.Workplane("XY").box(20, 20, 20)
result = box1.intersect(box2)
cq.exporters.export(result, 'GT.stl')