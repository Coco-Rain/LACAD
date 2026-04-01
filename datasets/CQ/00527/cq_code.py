import cadquery as cq

result = (
cq.Workplane("XZ")
.hLine(10, forConstruction=True)
.moveTo(10, 0)
.vLine(3)
.hLine(-4)
.mirrorX()
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')