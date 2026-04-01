import cadquery as cq

result = (
cq.Workplane("XZ")
.moveTo(-4, 0)
.hLine(3)
.vLine(6, True)
.hLine(5)
.moveTo(-1, 6)
.circle(1.5)
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')