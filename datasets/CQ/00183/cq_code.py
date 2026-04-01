import cadquery as cq

result = (
cq.Workplane("XY")
.circle(10).extrude(5)
.faces(">Z").workplane()
.polygon(6, 8).extrude(4)
.shell(1)
)
result
cq.exporters.export(result, 'GT.stl')