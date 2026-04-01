import cadquery as cq

result = (
cq.Workplane("XY")
.circle(12).extrude(8)
.faces(">Z").workplane()
.polygon(5, 8).extrude(4)
.faces(">Z[-2]").workplane()
.hole(6)
)
cq.exporters.export(result, 'GT.stl')