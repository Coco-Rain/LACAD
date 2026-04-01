import cadquery as cq

box = cq.Workplane("XY").box(1, 1, 1)
cylinder = cq.Workplane("XY").cylinder(0.5, 2)
result = (
cq.Workplane("XY")
.add(box)
.add(cylinder)
)
cq.exporters.export(result, 'GT.stl')