import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 10)
.shells("<Z")
)
cq.exporters.export(result, 'GT.stl')