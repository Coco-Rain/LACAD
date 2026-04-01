import cadquery as cq

result = (
cq.Workplane("XY")
.hLine(8)
.vLine(5)
.hLine(-8)
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')