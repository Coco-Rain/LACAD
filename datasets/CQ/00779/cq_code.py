import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 20)
.shells(">Z")
)
cq.exporters.export(result, 'GT.stl')